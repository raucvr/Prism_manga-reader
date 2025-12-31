"""
PDF Parser Service
解析 PDF 文档，提取文本和图像
"""

import io
import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, BinaryIO, Union

import pdfplumber
from PIL import Image


@dataclass
class ExtractedImage:
    """提取的图像"""
    page_number: int
    image_index: int
    data_base64: str
    mime_type: str = "image/png"
    width: int = 0
    height: int = 0
    caption: str = ""


@dataclass
class ExtractedPage:
    """提取的页面内容"""
    page_number: int
    text: str
    images: list[ExtractedImage] = field(default_factory=list)
    tables: list[list[list[str]]] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """解析后的文档"""
    filename: str
    total_pages: int
    pages: list[ExtractedPage]
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """获取完整文本"""
        return "\n\n".join(page.text for page in self.pages)

    @property
    def all_images(self) -> list[ExtractedImage]:
        """获取所有图像"""
        images = []
        for page in self.pages:
            images.extend(page.images)
        return images

    def get_text_chunks(self, max_tokens: int = 4000, overlap: int = 200) -> list[str]:
        """
        将文档分割成适合 LLM 处理的文本块

        Args:
            max_tokens: 每块最大 token 数（粗略估算，1 token ≈ 4 字符）
            overlap: 块之间的重叠字符数

        Returns:
            文本块列表
        """
        full_text = self.full_text
        max_chars = max_tokens * 4

        if len(full_text) <= max_chars:
            return [full_text]

        chunks = []
        start = 0

        while start < len(full_text):
            end = start + max_chars

            # 尝试在句子边界处分割
            if end < len(full_text):
                # 找最近的句子结束符
                for sep in ["。", ".", "！", "!", "？", "?", "\n\n"]:
                    last_sep = full_text.rfind(sep, start, end)
                    if last_sep > start:
                        end = last_sep + 1
                        break

            chunks.append(full_text[start:end].strip())
            start = end - overlap

        return chunks


class PDFParser:
    """PDF 解析器"""

    def __init__(self):
        self.min_image_size = 100  # 最小图像尺寸（像素）

    async def parse(
        self,
        source: Union[str, Path, BinaryIO],
        extract_images: bool = True,
        extract_tables: bool = True
    ) -> ParsedDocument:
        """
        解析 PDF 文档

        Args:
            source: PDF 文件路径或二进制流
            extract_images: 是否提取图像
            extract_tables: 是否提取表格

        Returns:
            ParsedDocument 对象
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            filename = path.name
        else:
            filename = "uploaded.pdf"

        pages = []

        with pdfplumber.open(source) as pdf:
            total_pages = len(pdf.pages)
            metadata = pdf.metadata or {}

            for page_num, page in enumerate(pdf.pages, start=1):
                extracted_page = await self._extract_page(
                    page,
                    page_num,
                    extract_images,
                    extract_tables
                )
                pages.append(extracted_page)

        return ParsedDocument(
            filename=filename,
            total_pages=total_pages,
            pages=pages,
            metadata=metadata
        )

    async def _extract_page(
        self,
        page,
        page_number: int,
        extract_images: bool,
        extract_tables: bool
    ) -> ExtractedPage:
        """提取单页内容"""
        # 提取文本
        text = page.extract_text() or ""

        # 提取图像
        images = []
        if extract_images:
            images = await self._extract_images(page, page_number)

        # 提取表格
        tables = []
        if extract_tables:
            raw_tables = page.extract_tables() or []
            tables = [
                [[cell or "" for cell in row] for row in table]
                for table in raw_tables
            ]

        return ExtractedPage(
            page_number=page_number,
            text=text,
            images=images,
            tables=tables
        )

    async def _extract_images(
        self,
        page,
        page_number: int
    ) -> list[ExtractedImage]:
        """提取页面中的图像"""
        images = []

        try:
            page_images = page.images or []
        except Exception:
            return images

        for idx, img_info in enumerate(page_images):
            try:
                # 检查图像尺寸
                width = int(img_info.get("width", 0))
                height = int(img_info.get("height", 0))

                if width < self.min_image_size or height < self.min_image_size:
                    continue

                # 获取图像数据
                x0 = img_info.get("x0", 0)
                y0 = img_info.get("top", 0)
                x1 = img_info.get("x1", width)
                y1 = img_info.get("bottom", height)

                # 裁剪页面获取图像区域
                cropped = page.within_bbox((x0, y0, x1, y1))
                if cropped:
                    img_page = cropped.to_image(resolution=150)
                    img_buffer = io.BytesIO()
                    img_page.original.save(img_buffer, format="PNG")
                    img_data = base64.b64encode(img_buffer.getvalue()).decode()

                    images.append(ExtractedImage(
                        page_number=page_number,
                        image_index=idx,
                        data_base64=img_data,
                        width=width,
                        height=height
                    ))

            except Exception as e:
                # 跳过无法提取的图像
                continue

        return images

    async def extract_figure_with_context(
        self,
        document: ParsedDocument,
        page_number: int,
        image_index: int,
        context_chars: int = 500
    ) -> tuple[ExtractedImage, str]:
        """
        提取图像及其周围的上下文文本

        Args:
            document: 解析后的文档
            page_number: 页码
            image_index: 图像索引
            context_chars: 上下文字符数

        Returns:
            (图像, 上下文文本) 元组
        """
        page = document.pages[page_number - 1]
        image = page.images[image_index]

        # 获取该页面的文本作为上下文
        context = page.text[:context_chars]

        return image, context


# 全局解析器实例
_parser: Optional[PDFParser] = None


def get_parser() -> PDFParser:
    """获取 PDF 解析器实例"""
    global _parser
    if _parser is None:
        _parser = PDFParser()
    return _parser
