import numpy as np
import streamlit as st

from image_generator.cell_spotter import preprocess_image, get_contours, draw_contours
from image_generator.tiff_display_utills import get_tiff_with_meta_data, display_im_with_cmap


# def prep_im_callback(im, value_range):
#
#     return prep_im


def preprocess_image_page(im, value_range, lut):
    threshold = st.number_input("Intensity threshold (add explanation):",
                                min_value=0, max_value=256, step=1)
    # go_btn = st.button("PREPROCESS")
    # if go_btn:
    prep_im = preprocess_image(im, value_range, threshold)
    st.image(prep_im)
    return prep_im


def get_contours_page(orig_im, prep_im, range, lut):
    min_area = st.number_input("Contour Area minimum:", min_value=0, step=1)
    contours = get_contours(prep_im, min_area)
    st.subheader(f"Found {len(contours)} cells!")
    st.write(prep_im.shape)
    im_cont = display_im_with_cmap(prep_im, lut, range)
    im_cont = draw_contours(im_cont, contours, thick=1)
    st.image(im_cont/256)


def main():
    # todo use state
    im = None
    prep_im = None
    st.title("CCFP-Cell Counter for Puchka")
    st.title("v0.0.1")
    st.subheader("Made with :heart:")
    st.text("upload your TIFF file")
    with st.expander("UPLOAD"):
        im = st.file_uploader('here', type=['tiff', 'tif'],)
    if im:
        t, ranges, luts = get_tiff_with_meta_data(im)
        i = st.slider("Choose first channel", min_value=0, max_value=t.shape[0] - 1, step=1)
        ch_im, value_range, lut = t[i, :, :], ranges[i * 2:i * 2 + 2], luts[i]
        pallete = np.linspace(value_range[0], value_range[1], 1000).astype(int)
        st.caption("this is the color of the channel you chose:")
        st.image(display_im_with_cmap(np.vstack([pallete, pallete]), lut, value_range))
        st.image(display_im_with_cmap(t[i, :, :], range=ranges[i * 2:i * 2 + 2], lut=luts[i]))
        with st.expander("Preprocess"):
            prep_im = preprocess_image_page(ch_im, value_range, lut)
        if prep_im is not None:
            get_contours_page(ch_im, prep_im, value_range, lut)
            st.text("HEY")


if __name__ == '__main__':
    main()
