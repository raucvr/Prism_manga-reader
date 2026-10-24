# Third-party notices

Prism's own code is licensed under the Apache License 2.0 (see `LICENSE`). The third-party files bundled with it keep
their own licenses, listed here.

## Fonts

Prism bundles these fonts in `backend/assets/fonts/`, so every operating system letters the manga with the same files.
Each file is the unmodified upstream release. `backend/assets/fonts/fonts.json` records each file's source and SHA-256,
and `backend/services/fonts.py` checks them once per run. The full license texts are in `backend/assets/fonts/LICENSES/`.

| File | Font and version | Used for | License | Copyright |
|---|---|---|---|---|
| `ChillRoundM.otf` | ChillRound M 1.805, derived from Zen Maru Gothic | Chinese text | SIL Open Font License 1.1, Reserved Font Names "ChillRoundF" and "ChillRoundM" | Copyright 2021 The Zen Maru Gothic Authors; © 2023 ChillType |
| `Shantell_Sans-Normal-Medium.otf`, `Shantell_Sans-Normal-Bold.otf` | Shantell Sans 1.011 (static "Normal" styles) | English text | SIL Open Font License 1.1 | Copyright 2022 The Shantell Sans Project Authors |
| `AtkinsonHyperlegibleNext-Regular.ttf`, `AtkinsonHyperlegibleNext-Bold.ttf` | Atkinson Hyperlegible Next 2.001 | digits in Chinese text, Latin fallback | SIL Open Font License 1.1 | Copyright 2020-2024 The Atkinson Hyperlegible Next Project Authors |
| `DejaVuSans.ttf` | DejaVu Sans 2.37 | symbols, Greek, superscripts and subscripts | Bitstream Vera Fonts license and Arev Fonts license; DejaVu's changes are in the public domain | Copyright (c) 2003 Bitstream, Inc.; Copyright (c) 2006 Tavmjong Bah |

The web reader bundles one more font, so building the frontend never fetches Google Fonts:

| File | Font and version | Used for | License | Copyright |
|---|---|---|---|---|
| `frontend/public/fonts/InterVariable-latin.woff2` | Inter 4, variable, Latin subset (from `@fontsource-variable/inter` 5.3.0) | the web interface | SIL Open Font License 1.1 (`frontend/public/fonts/OFL.txt`) | Copyright 2016 The Inter Project Authors |

SHA-256 of `InterVariable-latin.woff2`: `3100e775e8616cd2611beecfa23a4263d7037586789b43f035236a2e6fbd4c62`.

License files of the manga fonts:

- ChillRound M: `LICENSES/OFL-ChillRound.txt`
- Shantell Sans: `LICENSES/OFL-ShantellSans.txt`
- Atkinson Hyperlegible Next: `LICENSES/OFL-AtkinsonHyperlegibleNext.txt`
- DejaVu Sans: `LICENSES/LICENSE-DejaVu.txt`

Sources and SHA-256 of the bundled files:

| File | Downloaded from | SHA-256 |
|---|---|---|
| `ChillRoundM.otf` | https://github.com/Warren2060/ChillRound/releases/download/v1.805/ChillRoundM_v1.805.zip (archive SHA-256 `197018fdce5436fc99d8874d32c5b4702b2cc595006d2d341fb3542b6b5779bd`, as GitHub publishes it) | `2c8da065414e3cab951744f15c25b17a9ab4ae93e48def22004bc917550d1219` |
| `Shantell_Sans-Normal-Medium.otf` | https://github.com/arrowtype/shantell-sans/releases/download/1.011/Shantell_Sans_1.011.zip (`Desktop/Static/`) | `97faf292b154159d61a321e48116fd05844c65b5433f6a3c05690c9fe966456b` |
| `Shantell_Sans-Normal-Bold.otf` | same archive | `7b446a4d2f0e31d08692339cf019b3321e612a6dd28bd679979fad809548ed2c` |
| `AtkinsonHyperlegibleNext-Regular.ttf` | https://github.com/googlefonts/atkinson-hyperlegible-next, commit `7925f50f649b3813257faf2f4c0b381011f434f1`, `fonts/ttf/` | `88ed5c31a71584c7772963b02d04bef1eb7e3d2e9c8b9cb204339b1f82cf432c` |
| `AtkinsonHyperlegibleNext-Bold.ttf` | same commit | `994414047df66bb4998d01c1cb1eeb4a2ddc4622d1aa56bbb8adbeca7645b041` |
| `DejaVuSans.ttf` | https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-sans-ttf-2.37.zip (`ttf/`) | `7da195a74c55bef988d0d48f9508bd5d849425c1770dba5d7bfc6ce9ed848954` |

Notes:

- Shantell Sans and Atkinson Hyperlegible Next declare no Reserved Font Name. ChillRound M reserves its names, so it is
  bundled unmodified (a subset or any other modified version would have to be renamed). The OFL allows bundling and
  redistributing these fonts with software, provided the copyright notice and license go with them and the fonts are
  not sold by themselves.
- The DejaVu Sans license requires its copyright and permission notice to go with the font. A modified copy must not
  use the names "Bitstream" or "Vera", or "Arev" for the glyphs taken from the Arev fonts.
