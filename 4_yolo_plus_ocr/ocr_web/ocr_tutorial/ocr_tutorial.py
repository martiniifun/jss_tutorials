import reflex as rx
import numpy as np
import platform
from PIL import ImageFont, ImageDraw, Image
from matplotlib import pyplot as plt

import uuid
import json
import time
import cv2
import requests


def get_ocr_result(path):
    files = [
        ("file", open(path, "rb"))
    ]
    api_url = "https://9upvb67beq.apigw.ntruss.com/custom/v1/28024/7cb9fc59c2d43289019b8324b342505401dcaebbc780f57c7030a8fa4a1afa5d/general"
    secret_key = "T0FsWlRPdGZ4TUxHd2phUVd2cHduQklWU2t5cVRzUnU="
    request_json = {
        'images': [
            {
                'format': path.split(".")[-1],
                'name': path.split(".")[0]
            }
        ],
        'requestId': str(uuid.uuid4()),
        'version': "V2",
        'timestamp': int(round(time.time() * 1000)),
    }
    payload = {
        "message": json.dumps(request_json).encode("utf-8")
    }
    headers = {
        'X-OCR-SECRET': secret_key,
    }
    response = requests.request("POST", api_url,
                                headers=headers,
                                data=payload,
                                files=files)
    result = response.json()
    field_list = result["images"][0]["fields"]
    result = "|".join([i["inferText"] for i in field_list])
    return result


class State(rx.State):
    """The app state."""

    # The images to show.
    img: list[str]
    result_text = "result"

    async def handle_upload(
            self, files: list[rx.UploadFile]
    ):
        """Handle the upload of file(s).

        Args:
            files: The uploaded files.
        """
        for file in files:
            upload_data = await file.read()
            outfile = rx.get_asset_path(file.filename)
            print("outfile: ", outfile)

            # Save the file.
            with open(outfile, "wb") as file_object:
                file_object.write(upload_data)

            # Update the img var.
            self.img.append(file.filename)
        self.result_text = get_ocr_result(outfile)


color = "rgb(107,99,246)"


def index():
    """The main view."""
    return rx.vstack(
        rx.upload(
            rx.vstack(
                rx.button(
                    "Select File",
                    color=color,
                    bg="white",
                    border=f"1px solid {color}",
                ),
                rx.text(
                    "Drag and drop files here or click to select files"
                ),
            ),
            border=f"1px dotted {color}",
            padding="5em",
        ),
        rx.hstack(rx.foreach(rx.selected_files, rx.text)),
        rx.button(
            "Upload",
            on_click=lambda: State.handle_upload(
                rx.upload_files()
            ),
        ),
        rx.button(
            "Clear",
            on_click=rx.clear_selected_files,
        ),
        rx.foreach(
            State.img, lambda img: rx.image(src=img)
        ),
        rx.text(State.result_text),
        padding="5em",
    )


# 앱 생성
app = rx.App()
# 페이지 추가
app.add_page(index)
