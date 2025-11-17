import os
import logging
import io as python_io
from fractions import Fraction
from typing_extensions import override
import asyncio

import openai
import decord
import torch
import av

import folder_paths
from comfy_api.latest import ComfyExtension, io, IO
from comfy_api.latest._io import UploadType, FolderType
from comfy_api_nodes.util.conversions import pil_to_bytesio, tensor_to_pil
from comfy_api.input_impl import VideoFromComponents
from comfy_api.util import VideoComponents


AnimonIO_ApiKey = IO.Custom("ANIMON_KEY")
AnimonIO_ImageID = IO.Custom("ANIMON_IMAGE_ID")
AnimonIO_VideoID = IO.Custom("ANIMON_VIDEO_ID")
AnimonIO_Bytes = IO.Custom("ANIMON_BYTES")


#region key
class AnimonKeyNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AnimonKeyNode",
            display_name="Key",
            category="Animon",
            inputs=[
                io.String.Input(
                    "api_key", 
                    multiline=False,
                    tooltip="Your Animon API key, available from https://platform.openai.com/api-keys",
                ),
            ],
            outputs=[
                AnimonIO_ApiKey.Output("animon_key"),
            ],
        )
    
    @classmethod
    def execute(cls, api_key) -> io.NodeOutput:
        base_url = "https://platform.animon.ai/api/openai/v1/"
        animon_key = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        return io.NodeOutput(animon_key)


#region upload
class AnimonUploadImageFromFileNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])

        return io.Schema(
            node_id="AnimonUploadImageFromFileNode",
            display_name="Upload Image",
            category="Animon/Upload",
            inputs=[
                AnimonIO_ApiKey.Input("animon_key"),
                io.Combo.Input(
                    "image",
                    options=files,
                    upload=UploadType.image,
                    image_folder=FolderType.input,
                )
            ],
            outputs=[
                AnimonIO_ImageID.Output(
                    "image_id",
                    display_name="IMAGE_ID",
                ),
            ],
        )
    
    @classmethod
    def execute(cls, animon_key: openai.OpenAI, image: str) -> io.NodeOutput:
        input_dir = folder_paths.get_input_directory()
        image_path = os.path.join(input_dir, image)

        upload_file = animon_key.files.create(
            file=open(image_path, "rb"),
            purpose="video_generation"
        )

        return io.NodeOutput(upload_file.id)


class AnimonUploadVideoFromFileNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["video"])

        return io.Schema(
            node_id="AnimonUploadVideoFromFileNode",
            display_name="Upload Video",
            category="Animon/Upload",
            inputs=[
                AnimonIO_ApiKey.Input("animon_key"),
                io.Combo.Input(
                    "video",
                    options=files,
                    upload=UploadType.video,
                    image_folder=FolderType.input,
                )
            ],
            outputs=[
                AnimonIO_VideoID.Output(
                    "video_id",
                    display_name="VIDEO_ID",
                ),
            ],
        )
    
    @classmethod
    def execute(cls, animon_key: openai.OpenAI, video: str) -> io.NodeOutput:
        input_dir = folder_paths.get_input_directory()
        video_path = os.path.join(input_dir, video)

        upload_file = animon_key.files.create(
            file=open(video_path, "rb"),
            purpose="video_generation"
        )

        return io.NodeOutput(upload_file.id)


def image_tensor_to_named_png_bytes(image: torch.Tensor, name: str) -> python_io.BytesIO:
    # image from tensor to bytes
    pil_image = tensor_to_pil(image, total_pixels=6000 * 6000)
    buffer = pil_to_bytesio(pil_image, mime_type="image/png")
    buffer.name = name
    return buffer


def image_tensor_to_png_bytes(image: torch.Tensor) -> bytes:
    # image from tensor to bytes
    img_bytes_io = image_tensor_to_named_png_bytes(image, name="image.png")
    img_bytes = img_bytes_io.getvalue()
    return img_bytes


def video_tensor_to_named_mp4_bytes(video: torch.Tensor, name: str) -> python_io.BytesIO:
    # tensor shape FWHC to FCWH
    if video.dim() == 4 and video.shape[-1] == 3:
        video = video.permute(0, 3, 1, 2)

    # convert to numpy and scale to 0-255
    video_np = (video.permute(0, 2, 3, 1).numpy() * 255).astype('uint8')

    # in-memory buffer
    buffer = python_io.BytesIO()
    buffer.name = name
    
    # encode video using av
    with av.open(buffer, mode='w', format='mp4') as container:
        stream = container.add_stream('libx264', rate=30)
        stream.width = video_np.shape[2]
        stream.height = video_np.shape[1]
        stream.pix_fmt = 'yuv420p'
        
        for frame_np in video_np:
            frame = av.VideoFrame.from_ndarray(frame_np, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        
        # Flush remaining frames
        for packet in stream.encode():
            container.mux(packet)
    
    buffer.seek(0)
    return buffer


def video_tensor_to_mp4_bytes(video: torch.Tensor) -> bytes:
    # video from tensor to bytes
    video_bytes_io = video_tensor_to_named_mp4_bytes(video, name="video.mp4")
    video_bytes = video_bytes_io.getvalue()
    return video_bytes


class AnimonUploadImageFromTensorNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AnimonUploadImageFromTensorNode",
            display_name="Upload Image (from Tensor)",
            category="Animon/Upload",
            inputs=[
                AnimonIO_ApiKey.Input("animon_key"),
                io.Image.Input("image"),
            ],
            outputs=[
                AnimonIO_ImageID.Output(
                    "image_id",
                    display_name="IMAGE_ID",
                ),
            ],
        )
    
    @classmethod
    def execute(cls, animon_key: openai.OpenAI, image: torch.Tensor) -> io.NodeOutput:
        image_buffer = image_tensor_to_named_png_bytes(image, "image.png")

        upload_file = animon_key.files.create(
            file=image_buffer,
            purpose="video_generation"
        )

        return io.NodeOutput(upload_file.id)


class AnimonUploadVideoFromTensorNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AnimonUploadVideoFromTensorNode",
            display_name="Upload Video (from Tensor)",
            category="Animon/Upload",
            inputs=[
                AnimonIO_ApiKey.Input("animon_key"),
                io.Video.Input("video"),
            ],
            outputs=[
                AnimonIO_VideoID.Output(
                    "video_id",
                    display_name="VIDEO_ID",
                ),
            ],
        )
    
    @classmethod
    def execute(cls, animon_key: openai.OpenAI, video: torch.Tensor) -> io.NodeOutput:
        # video from tensor to bytes
        video_buffer = video_tensor_to_named_mp4_bytes(video, "video.mp4")

        upload_file = animon_key.files.create(
            file=video_buffer,
            purpose="video_generation"
        )

        return io.NodeOutput(upload_file.id)


class AnimonUploadVideoFromBytesNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AnimonUploadVideoFromBytesNode",
            display_name="Upload Video (from Bytes)",
            category="Animon/Upload",
            inputs=[
                AnimonIO_ApiKey.Input("animon_key"),
                AnimonIO_Bytes.Input("video_bytes"),
            ],
            outputs=[
                AnimonIO_VideoID.Output(
                    "video_id",
                    display_name="VIDEO_ID",
                ),
            ],
        )
    
    @classmethod
    def execute(cls, animon_key: openai.OpenAI, video_bytes: bytes) -> io.NodeOutput:
        buffer = python_io.BytesIO(video_bytes)
        buffer.name = "video.mp4"
        
        upload_file = animon_key.files.create(
            file=buffer,
            purpose="video_generation"
        )

        return io.NodeOutput(upload_file.id)


#region generate
async def wait_for_video_completion(animon_key: openai.OpenAI, video_id: str) -> openai.types.video.Video:
    video = animon_key.videos.retrieve(video_id)
    while True:
        await asyncio.sleep(10)

        video = animon_key.videos.retrieve(video_id)
        if video.status == "in_progress" or video.status == "queued":
            continue
        elif video.status == "completed":
            break
        else:
            raise RuntimeError(f"Video generation failed: {video}")

    return video


def parse_video_from_animon(response: openai._legacy_response.HttpxBinaryResponseContent) -> torch.Tensor:
    # use decord with the buffer directly
    video_buffer = python_io.BytesIO(response.content)
    vr = decord.VideoReader(video_buffer)
    fps = vr.get_avg_fps()
    frames = vr.get_batch(list(range(len(vr)))).asnumpy()
    video_tensor = torch.from_numpy(frames).float() / 255.0

    return video_tensor, fps


class AnimonImageToVideoNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AnimonImageToVideoNode",
            display_name="Image to Video",
            category="Animon",
            inputs=[
                AnimonIO_ApiKey.Input("animon_key"),
                io.Combo.Input(
                    "model",
                    options=["anicut-1-5", "anicut-pro-1-5", "anicut-pro-1-6"],
                    default="anicut-1-5",
                ),
                io.Image.Input("image"),
                io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                ),
                io.Combo.Input(
                    "resolution",
                    options=["480P", "720P", "1080P"],
                    default="480P",
                ),
                io.Int.Input(
                    "seed",
                    min=0, max=2147483647, step=1, default=42,
                    control_after_generate=True,
                    display_mode=IO.NumberDisplay.number,
                ),
                io.Int.Input(
                    "num_frames",
                    min=17, max=81, step=4, default=81,
                    display_mode=IO.NumberDisplay.number,
                )
            ],
            outputs=[
                io.Video.Output(
                    "video",
                    display_name="VIDEO",
                ),
                AnimonIO_Bytes.Output(
                    "video_bytes",
                    display_name="VIDEO_BYTES",
                ),
            ],
        )
    
    @classmethod
    async def execute(cls, animon_key: openai.OpenAI, model: str, image: torch.Tensor, 
                      prompt: str, resolution: str, seed: int, num_frames: int) -> io.NodeOutput:
        # image from tensor to bytes
        pil_image = tensor_to_pil(image, total_pixels=6000 * 6000)
        img_byte_arr = pil_to_bytesio(pil_image, mime_type="image/png")
        img_bytes = img_byte_arr.getvalue()

        # call api
        task = animon_key.videos.create(
            model=model, prompt=prompt, input_reference=img_bytes,
            extra_body={
                "resolution": resolution,
                "seed": seed,
                "num_frames": num_frames,
            }
        )
        logging.info(f"[AnimonI2V] start generating, video ID: {task.id}")

        # wait for completion
        video = await wait_for_video_completion(animon_key, task.id)

        # download content
        video_id = video.id
        response = animon_key.videos.download_content(video_id)
        video_tensor, fps = parse_video_from_animon(response)
        video_output = VideoFromComponents(VideoComponents(images=video_tensor, frame_rate=Fraction(fps)))

        print(f"response content size: {len(response.content)} bytes, type: {type(response.content)}")

        return io.NodeOutput(video_output, response.content)


class AnimonStartEndToVideoNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AnimonStartEndToVideoNode",
            display_name="Start-End to Video",
            category="Animon",
            inputs=[
                AnimonIO_ApiKey.Input("animon_key"),
                io.Combo.Input(
                    "model",
                    options=["anicut-ib-1-5", "anicut-pro-ib-1-5", "anicut-pro-ib-1-6"],
                    default="anicut-ib-1-5",
                ),
                AnimonIO_ImageID.Input("image_start"),
                AnimonIO_ImageID.Input("image_end"),
                io.String.Input(
                    "prompt",
                    display_name="prompt",
                    default="",
                    multiline=True,
                ),
                io.Combo.Input(
                    "resolution",
                    display_name="resolution",
                    options=["480P", "720P", "1080P"],
                    default="480P",
                ),
                io.Int.Input(
                    "seed",
                    display_name="seed",
                    min=0, max=2147483647, step=1,
                    default=42,
                    control_after_generate=True,
                    display_mode=IO.NumberDisplay.number,
                ),
                io.Int.Input(
                    "num_frames",
                    min=17, max=81, step=4,
                    default=81,
                    display_mode=IO.NumberDisplay.number,
                )
            ],
            outputs=[
                io.Video.Output(
                    "video",
                    display_name="VIDEO",
                ),
                AnimonIO_Bytes.Output(
                    "video_bytes",
                    display_name="VIDEO_BYTES",
                ),
            ],
        )
    
    @classmethod
    async def execute(cls, animon_key: openai.OpenAI, model: str, image_start: str, image_end: str, 
                      prompt: str, resolution: str, seed: int, num_frames: int) -> io.NodeOutput:
        # call api
        task = animon_key.videos.create(
            model=model, prompt=prompt,
            extra_body={
                "image_start": image_start,  # Use the uploaded start image ID
                "image_end": image_end,      # Use the uploaded end image ID
                "resolution": resolution,
                "seed": seed,
                "num_frames": num_frames,
            }
        )
        logging.info(f"[AnimonIB2V] start generating, video ID: {task.id}")

        # wait for completion
        video = await wait_for_video_completion(animon_key, task.id)

        # download content
        video_id = video.id
        response = animon_key.videos.download_content(video_id)
        video_tensor, fps = parse_video_from_animon(response)
        video_output = VideoFromComponents(VideoComponents(images=video_tensor, frame_rate=Fraction(fps)))

        return io.NodeOutput(video_output, response.content)


class AnimonUpscaleVideoNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AnimonUpscaleVideoNode",
            display_name="Upscale Video",
            category="Animon",
            inputs=[
                AnimonIO_ApiKey.Input("animon_key"),
                AnimonIO_VideoID.Input("video_id"),
                io.Combo.Input(
                    "model",
                    display_name="model",
                    options=["anisr"],
                    default="anisr",
                ),
                io.Combo.Input(
                    "resolution",
                    display_name="resolution",
                    options=["1080P"],
                    default="1080P",
                ),
            ],
            outputs=[
                io.Video.Output(
                    "upscaled_video",
                    display_name="VIDEO",
                ),
                AnimonIO_Bytes.Output(
                    "video_bytes",
                    display_name="VIDEO_BYTES",
                ),
            ],
        )
    
    @classmethod
    async def execute(cls, animon_key: openai.OpenAI, video_id: str, model: str, resolution: str) -> io.NodeOutput:
        # call api
        task = animon_key.videos.create(
            model=model, prompt="",
            extra_body={
                "video_id": video_id,
                "resolution": resolution,
            }
        )
        logging.info(f"[AnimonUpscale] start generating, video ID: {task.id}")

        # wait for completion
        video = await wait_for_video_completion(animon_key, task.id)

        # download content
        video_id = video.id
        response = animon_key.videos.download_content(video_id)
        video_tensor, fps = parse_video_from_animon(response)
        video_output = VideoFromComponents(VideoComponents(images=video_tensor, frame_rate=Fraction(fps)))

        return io.NodeOutput(video_output, response.content)


#region extension
class ComfyUIAnimonExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            AnimonKeyNode,

            AnimonUploadImageFromFileNode,
            AnimonUploadVideoFromFileNode,
            AnimonUploadImageFromTensorNode,
            AnimonUploadVideoFromTensorNode,
            AnimonUploadVideoFromBytesNode,

            AnimonImageToVideoNode,
            AnimonStartEndToVideoNode,
            AnimonUpscaleVideoNode,
        ]


async def comfy_entrypoint() -> ComfyUIAnimonExtension:
    return ComfyUIAnimonExtension()


# NODE_CLASS_MAPPINGS for ComfyUI IO V3 compatibility
NODE_CLASS_MAPPINGS_V3 = {
    "AnimonKeyNode": AnimonKeyNode,

    "AnimonUploadImageFromFileNode": AnimonUploadImageFromFileNode,
    "AnimonUploadVideoFromFileNode": AnimonUploadVideoFromFileNode,
    "AnimonUploadImageFromTensorNode": AnimonUploadImageFromTensorNode,
    "AnimonUploadVideoFromTensorNode": AnimonUploadVideoFromTensorNode,
    "AnimonUploadVideoFromBytesNode": AnimonUploadVideoFromBytesNode,

    "AnimonImageToVideoNode": AnimonImageToVideoNode,
    "AnimonStartEndToVideoNode": AnimonStartEndToVideoNode,
    "AnimonUpscaleVideoNode": AnimonUpscaleVideoNode,
}
