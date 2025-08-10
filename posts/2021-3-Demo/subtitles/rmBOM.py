# remove_bom_vtt.py
import codecs

in_file = "GMT20250810-032335_Clip.vtt"
out_file = "fixed.vtt"

with codecs.open(in_file, "r", encoding="utf-8-sig") as f:  # utf-8-sig 会自动去掉 BOM
    content = f.read()

with codecs.open(out_file, "w", encoding="utf-8") as f:
    f.write(content)

print(f"已去掉 BOM，保存到 {out_file}")
