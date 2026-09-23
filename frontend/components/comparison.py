from __future__ import annotations

import base64
from html import escape
from io import BytesIO

import streamlit as st
from PIL import Image


def _image_to_base64(image: Image.Image) -> str:
    """Convert a PIL image to a base64 PNG data URL."""

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


def before_after_slider(
    before: Image.Image,
    after: Image.Image,
    before_label: str = "Original — 10 m",
    after_label: str = "Super Resolved — 2.5 m",
) -> None:
    """
    Display an interactive before/after comparison.

    LEFT  = before/original
    RIGHT = after/super-resolved
    """

    if before is None or after is None:
        raise ValueError(
            "Both before and after images are required."
        )

    before_label, after_label = escape(before_label), escape(after_label)
    before_data = _image_to_base64(before)
    after_data = _image_to_base64(after)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>

        <style>

            * {{
                box-sizing: border-box;
            }}

            html,
            body {{
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;

                font-family:
                    -apple-system,
                    BlinkMacSystemFont,
                    "Segoe UI",
                    sans-serif;

                background: transparent;
            }}

            .comparison {{
                position: relative;

                width: 100%;
                max-width: 1100px;

                height: 560px;

                margin: auto;

                overflow: hidden;

                border-radius: 14px;

                background: #111827;

                user-select: none;
            }}

            /*
             * BASE IMAGE
             *
             * Original image occupies the entire container.
             */
            .base-image {{
                position: absolute;

                inset: 0;

                width: 100%;
                height: 100%;

                object-fit: contain;

                background: #111827;
            }}

            /*
             * AFTER IMAGE CONTAINER
             *
             * The SR image is positioned on the RIGHT side.
             */
            .after-container {{
                position: absolute;

                top: 0;
                right: 0;

                width: 100%;
                clip-path: inset(0 0 0 50%);
                height: 100%;

                overflow: hidden;


                z-index: 5;
            }}

            /*
             * Make the SR image span the complete comparison
             * area while keeping its right-side position.
             */
            .after-image {{
                position: absolute;

                top: 0;
                right: 0;

                width: 100%;
                max-width: none;

                height: 100%;

                object-fit: contain;

                background: #111827;
            }}

            /*
             * LABELS
             */
            .label {{
                position: absolute;

                top: 16px;

                padding: 7px 12px;

                border-radius: 8px;

                color: white;

                background: rgba(0, 0, 0, 0.70);

                font-size: 14px;

                font-weight: 600;

                z-index: 20;

                white-space: nowrap;
            }}

            .before-label {{
                left: 16px;
            }}

            .after-label {{
                right: 16px;
            }}

            /*
             * SLIDER LINE
             */
            .slider {{
                position: absolute;

                top: 0;
                left: 50%;

                width: 3px;
                height: 100%;

                background: white;

                transform: translateX(-50%);

                z-index: 25;

                pointer-events: none;
            }}

            /*
             * SLIDER HANDLE
             */
            .slider-button {{
                position: absolute;

                top: 50%;
                left: 50%;

                width: 48px;
                height: 48px;

                transform: translate(
                    -50%,
                    -50%
                );

                border-radius: 50%;

                background: white;

                box-shadow:
                    0 3px 14px
                    rgba(0, 0, 0, 0.40);

                display: flex;

                align-items: center;
                justify-content: center;

                color: #111827;

                font-size: 20px;

                font-weight: 700;
            }}

            /*
             * INVISIBLE RANGE CONTROL
             */
            .range {{
                position: absolute;

                top: 0;
                left: 0;

                width: 100%;
                height: 100%;

                margin: 0;

                opacity: 0;

                cursor: ew-resize;

                z-index: 30;
            }}

        </style>

    </head>

    <body>

        <div class="comparison">

            <!-- =========================================
                 ORIGINAL IMAGE
                 ========================================= -->

            <img
                class="base-image"
                src="{before_data}"
                alt="Original Sentinel-2"
            />


            <!-- =========================================
                 SUPER-RESOLVED IMAGE
                 ========================================= -->

            <div
                class="after-container"
                id="afterContainer"
            >

                <img
                    class="after-image"
                    src="{after_data}"
                    alt="ESA LDSR-S2 super-resolved"
                />

            </div>


            <!-- =========================================
                 LABELS
                 ========================================= -->

            <div class="label before-label">
                {before_label}
            </div>

            <div class="label after-label">
                {after_label}
            </div>


            <!-- =========================================
                 SLIDER
                 ========================================= -->

            <div
                class="slider"
                id="slider"
            >

                <div class="slider-button">
                    ↔
                </div>

            </div>


            <!-- =========================================
                 RANGE INPUT
                 ========================================= -->

            <input
                class="range"
                id="range"
                type="range"
                min="0"
                max="100"
                value="50"
                aria-label="Before and after comparison"
            />

        </div>


        <script>

            const range =
                document.getElementById(
                    "range"
                );

            const afterContainer =
                document.getElementById(
                    "afterContainer"
                );

            const slider =
                document.getElementById(
                    "slider"
                );


            function updateSlider(value) {{

                /*
                 * RIGHT SIDE = SR IMAGE
                 *
                 * When value = 50:
                 *
                 * LEFT  50% → Original
                 * RIGHT 50% → SR
                 */

                const rightWidth =
                    100 - Number(value);

                afterContainer.style.clipPath =
                    "inset(0 0 0 " + value + "%)";

                slider.style.left =
                    value + "%";
            }}


            range.addEventListener(
                "input",
                function() {{

                    updateSlider(
                        this.value
                    );

                }}
            );


            updateSlider(50);

        </script>

    </body>
    </html>
    """

    st.components.v1.html(
        html,
        height=580,
        scrolling=False,
    )
