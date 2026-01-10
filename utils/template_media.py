import os
import tempfile
from PIL import Image

def system_instruction(content):
    if isinstance(content, list):
        text = " ".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text")
    else:
        text = str(content)
    
    text = text[:150]
    instruction = f"""Generate exactly 4 words to summarize what this text is talking about:

    {text}

    Reply with only 4 words as a title, no extra text.
    
    title : """
    
    return instruction


def build_prompt_template(messages, system_prompt, mode):
    system_content = [{"type": "text", "text":system_prompt}] if mode == "vlm" else system_prompt
    prompt = [{"role": "system", "content": system_content}]
    for msg in messages:
        if mode == "vlm":
            prompt.append({"role": msg.get("role"), "content": msg.get("content")})
        else:    
            prompt.append({"role": msg.get("role"), "content": msg.get("text")})
    return prompt


def media_resize(file, max_size=720):
        """Resize image or video and return temp path"""
        file_extension = os.path.splitext(file.name)[1]
        if file.type.startswith("image"):
            img = Image.open(file).convert('RGB')
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as f:
                img.save(f, format='JPEG', quality=85, optimize=True)
                return f.name
        
        elif file.type.startswith("video"):
            from moviepy.editor import VideoFileClip
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as f:
                f.write(file.read())
                temp_input = f.name
            
            clip = VideoFileClip(temp_input)
            if max(clip.h, clip.w) > max_size:
                clip = clip.resize(height=max_size) if clip.h > clip.w else clip.resize(width=max_size)
            
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as f:
                output_path = f.name
            
            clip.write_videofile(output_path, codec='libx264', audio_codec='aac', verbose=False, logger=None)
            clip.close()
            return output_path