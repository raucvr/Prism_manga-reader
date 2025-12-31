"""
Dialogue Renderer Service
对白气泡渲染模块 - 在漫画图像上添加对白文字

支持功能:
1. 圆角气泡绘制
2. 多角色对白（不同位置）
3. 中文文字自动换行
4. 可爱风格的气泡设计
"""

import io
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


@dataclass
class BubbleStyle:
    """气泡样式配置"""
    bg_color: str = "#FFFFFF"
    border_color: str = "#333333"
    border_width: int = 3
    text_color: str = "#333333"
    font_size: int = 24
    padding: int = 15
    corner_radius: int = 20
    tail_size: int = 15  # 气泡尾巴大小
    max_width: int = 280  # 气泡最大宽度
    line_spacing: int = 6  # 行间距


class DialogueRenderer:
    """对白渲染器"""

    # 中文字体路径（按优先级）
    FONT_PATHS = [
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",     # 黑体
        "C:/Windows/Fonts/simsun.ttc",     # 宋体
    ]

    def __init__(self):
        self.font_path = self._find_font()
        self.default_style = BubbleStyle()

    def _find_font(self) -> str:
        """查找可用的中文字体"""
        for path in self.FONT_PATHS:
            if Path(path).exists():
                return path
        # 回退到默认字体
        return None

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """获取指定大小的字体"""
        if self.font_path:
            return ImageFont.truetype(self.font_path, size)
        return ImageFont.load_default()

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        """
        中文文字自动换行

        Args:
            text: 原始文本
            font: 字体对象
            max_width: 最大宽度

        Returns:
            换行后的文本列表
        """
        lines = []
        current_line = ""

        for char in text:
            test_line = current_line + char
            bbox = font.getbbox(test_line)
            width = bbox[2] - bbox[0]

            if width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = char

        if current_line:
            lines.append(current_line)

        return lines

    def _calculate_bubble_size(
        self,
        lines: List[str],
        font: ImageFont.FreeTypeFont,
        style: BubbleStyle
    ) -> Tuple[int, int]:
        """计算气泡尺寸"""
        if not lines:
            return (0, 0)

        # 计算文本区域尺寸
        max_line_width = 0
        total_height = 0

        for line in lines:
            bbox = font.getbbox(line)
            line_width = bbox[2] - bbox[0]
            line_height = bbox[3] - bbox[1]
            max_line_width = max(max_line_width, line_width)
            total_height += line_height + style.line_spacing

        total_height -= style.line_spacing  # 去掉最后一行的间距

        # 加上内边距
        bubble_width = max_line_width + style.padding * 2
        bubble_height = total_height + style.padding * 2

        return (bubble_width, bubble_height)

    def _draw_rounded_rectangle(
        self,
        draw: ImageDraw.Draw,
        xy: Tuple[int, int, int, int],
        radius: int,
        fill: str,
        outline: str,
        width: int
    ):
        """绘制圆角矩形"""
        x1, y1, x2, y2 = xy

        # 使用 PIL 的圆角矩形（如果可用）
        try:
            draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
        except AttributeError:
            # 旧版 PIL 回退方案
            # 绘制主体矩形
            draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill)
            draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill)

            # 绘制四个角的圆
            draw.ellipse([x1, y1, x1 + radius * 2, y1 + radius * 2], fill=fill)
            draw.ellipse([x2 - radius * 2, y1, x2, y1 + radius * 2], fill=fill)
            draw.ellipse([x1, y2 - radius * 2, x1 + radius * 2, y2], fill=fill)
            draw.ellipse([x2 - radius * 2, y2 - radius * 2, x2, y2], fill=fill)

            # 绘制边框
            if width > 0:
                draw.arc([x1, y1, x1 + radius * 2, y1 + radius * 2], 180, 270, fill=outline, width=width)
                draw.arc([x2 - radius * 2, y1, x2, y1 + radius * 2], 270, 360, fill=outline, width=width)
                draw.arc([x1, y2 - radius * 2, x1 + radius * 2, y2], 90, 180, fill=outline, width=width)
                draw.arc([x2 - radius * 2, y2 - radius * 2, x2, y2], 0, 90, fill=outline, width=width)
                draw.line([x1 + radius, y1, x2 - radius, y1], fill=outline, width=width)
                draw.line([x1 + radius, y2, x2 - radius, y2], fill=outline, width=width)
                draw.line([x1, y1 + radius, x1, y2 - radius], fill=outline, width=width)
                draw.line([x2, y1 + radius, x2, y2 - radius], fill=outline, width=width)

    def _draw_bubble_tail(
        self,
        draw: ImageDraw.Draw,
        bubble_rect: Tuple[int, int, int, int],
        tail_direction: str,
        style: BubbleStyle
    ):
        """
        绘制气泡尾巴

        Args:
            draw: ImageDraw 对象
            bubble_rect: 气泡矩形 (x1, y1, x2, y2)
            tail_direction: 尾巴方向 ("bottom-left", "bottom-right", "bottom-center")
            style: 气泡样式
        """
        x1, y1, x2, y2 = bubble_rect
        tail_size = style.tail_size

        if tail_direction == "bottom-left":
            # 左下角尾巴
            points = [
                (x1 + 30, y2),
                (x1 + 20, y2 + tail_size),
                (x1 + 50, y2)
            ]
        elif tail_direction == "bottom-right":
            # 右下角尾巴
            points = [
                (x2 - 50, y2),
                (x2 - 20, y2 + tail_size),
                (x2 - 30, y2)
            ]
        else:
            # 底部中央尾巴
            cx = (x1 + x2) // 2
            points = [
                (cx - 10, y2),
                (cx, y2 + tail_size),
                (cx + 10, y2)
            ]

        # 绘制尾巴
        draw.polygon(points, fill=style.bg_color, outline=style.border_color)
        # 覆盖连接处的边框
        draw.line([(points[0][0] + 2, y2 - 1), (points[2][0] - 2, y2 - 1)],
                  fill=style.bg_color, width=style.border_width + 2)

    def render_dialogue_bubble(
        self,
        image: Image.Image,
        text: str,
        position: Tuple[int, int],
        character_name: str = "",
        style: Optional[BubbleStyle] = None,
        tail_direction: str = "bottom-center"
    ) -> Image.Image:
        """
        在图像上渲染对白气泡

        Args:
            image: 原始图像
            text: 对白文本
            position: 气泡位置 (x, y) - 气泡底部中心点
            character_name: 角色名称（可选，用于调试）
            style: 气泡样式
            tail_direction: 尾巴方向

        Returns:
            添加了对白气泡的图像
        """
        if not text or not text.strip():
            return image

        style = style or self.default_style
        img = image.copy()
        draw = ImageDraw.Draw(img)
        font = self._get_font(style.font_size)

        # 文字换行
        lines = self._wrap_text(text.strip(), font, style.max_width - style.padding * 2)

        if not lines:
            return image

        # 计算气泡尺寸
        bubble_width, bubble_height = self._calculate_bubble_size(lines, font, style)

        # 计算气泡位置（position 是底部中心点）
        x, y = position
        x1 = x - bubble_width // 2
        y1 = y - bubble_height - style.tail_size
        x2 = x1 + bubble_width
        y2 = y1 + bubble_height

        # 确保气泡在图像范围内
        if x1 < 10:
            offset = 10 - x1
            x1 += offset
            x2 += offset
        if x2 > img.width - 10:
            offset = x2 - (img.width - 10)
            x1 -= offset
            x2 -= offset
        if y1 < 10:
            # 如果顶部超出，改为在底部显示
            y1 = y + style.tail_size
            y2 = y1 + bubble_height
            tail_direction = "top-center"  # 尾巴朝上

        bubble_rect = (x1, y1, x2, y2)

        # 绘制气泡主体
        self._draw_rounded_rectangle(
            draw, bubble_rect,
            style.corner_radius,
            style.bg_color,
            style.border_color,
            style.border_width
        )

        # 绘制尾巴
        if not tail_direction.startswith("top"):
            self._draw_bubble_tail(draw, bubble_rect, tail_direction, style)

        # 绘制文字
        text_x = x1 + style.padding
        text_y = y1 + style.padding

        for line in lines:
            draw.text((text_x, text_y), line, fill=style.text_color, font=font)
            bbox = font.getbbox(line)
            line_height = bbox[3] - bbox[1]
            text_y += line_height + style.line_spacing

        return img

    def render_panel_dialogues(
        self,
        image: Image.Image,
        dialogues: Dict[str, str],
        characters: List[str]
    ) -> Image.Image:
        """
        为面板渲染所有角色的对白

        Args:
            image: 原始图像
            dialogues: 对白字典 {角色名: 对白内容}
            characters: 场景中的角色列表

        Returns:
            添加了所有对白的图像
        """
        if not dialogues:
            return image

        img = image.copy()
        width, height = img.size

        # 计算每个角色的气泡位置
        num_speakers = len(dialogues)

        # 预定义位置（根据说话者数量）
        if num_speakers == 1:
            positions = [(width // 2, height // 4)]
            tails = ["bottom-center"]
        elif num_speakers == 2:
            positions = [
                (width // 3, height // 4),
                (width * 2 // 3, height // 3)
            ]
            tails = ["bottom-left", "bottom-right"]
        else:
            # 3+ 角色，均匀分布
            positions = []
            tails = []
            for i in range(num_speakers):
                x = width * (i + 1) // (num_speakers + 1)
                y = height // 4 + (i % 2) * 50  # 交错高度
                positions.append((x, y))
                tails.append("bottom-center")

        # 渲染每个对白
        for i, (char_name, text) in enumerate(dialogues.items()):
            if i < len(positions):
                pos = positions[i]
                tail = tails[i]
            else:
                # 超出预定位置，使用默认
                pos = (width // 2, height // 4)
                tail = "bottom-center"

            img = self.render_dialogue_bubble(
                img, text, pos,
                character_name=char_name,
                tail_direction=tail
            )

        return img


# 全局实例
_renderer: Optional[DialogueRenderer] = None


def get_dialogue_renderer() -> DialogueRenderer:
    """获取对白渲染器实例"""
    global _renderer
    if _renderer is None:
        _renderer = DialogueRenderer()
    return _renderer
