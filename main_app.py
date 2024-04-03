import const

import time
from omegaconf import OmegaConf
import streamlit as st
st.set_page_config(**const.SET_PAGE_CONFIG)
st.markdown(const.HIDE_ST_STYLE, unsafe_allow_html=True)
from PIL import Image
import numpy as np
import pandas as pd

import glob
import torch
from transformers import CLIPImageProcessor
from src import lightning_model



def main():
    ''''''
    st.markdown(f'<h1 style="text-align:center;">GeoGuessrAI App</h1>', unsafe_allow_html=True)

    # Load the model cached
    @st.cache_resource
    def load_model():
        config = OmegaConf.load('config.yaml')
        model = lightning_model.BaseModel(config)
        state_dict = torch.hub.load_state_dict_from_url(const.MODEL_URL, map_location=torch.device('cuda'))
        model.load_state_dict(state_dict)

        processor = CLIPImageProcessor.from_pretrained(config.data.model_name)
        return model, processor
    model, processor = load_model()
    

    # Load test image (jgp)
    def scan_image_path():
        paths = []
        true_coords = []
        for path in glob.iglob(const.DATA_URL+'/*.jpg'):
            paths.append(path)
            coord_list = path.split("/")[-1].split(",")
            lat, lon = [float(coord_list[0]), float(coord_list[1][:-4])]
            true_coords.append((lat, lon))
        return paths, true_coords
    paths, true_coords = scan_image_path()

    ''''''
    
    #col1, col2 = st.columns([1, 1])
    img_source = st.radio('Image Source', ('Sample', 'Upload', 'None'), help='You can paste a street view image from clipboard or upload an image from your local machine.')
    if img_source == 'Sample':
        try:
            id = np.random.randint(0, len(paths))
            image_data = Image.open(paths[id])
        except:
            image_data = None
    # elif img_source == 'Paste':
    #     out = pbutton('Paste an image').image_data
    #     try:
    #         image_data = out.convert("RGB")
    #     except:
    #         image_data = None
    elif img_source == 'Upload':
        image_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
        try :
            image_data = Image.open(image_file)
        except:
            image_data = None
    else:
        image_data = None


    if image_data is not None:
        with st.spinner('Loading...'):
            start_time = time.time()
            st.image(image_data, caption='Uploaded image', use_column_width=True)
            image_data = processor(image_data, return_tensors="pt")['pixel_values']
            with torch.no_grad():
                coord = model(image_data).squeeze().cpu().numpy()
            
            if img_source == 'Sample':
                col1, col2 = st.columns([1, 2])
                col1.button('Next Street View Image', help='Click to see another sample image.')
                col2.success(f'Inference Time: {time.time()-start_time:.2f} [sec]')
            else:
                st.success(f'Inference Time: {time.time()-start_time:.2f} [sec]')

    else:
        st.info('Please upload an image.')
        coord = None


    
    # Display the result
    st.subheader('Result')
    if coord is not None:
        st.write(f'Predicted: {coord[0]:.4f}, {coord[1]:.4f}')
        st.write(f'True: {true_coords[id][0]:.4f}, {true_coords[id][1]:.4f}')
        st.map(pd.DataFrame({'lat': [coord[0], true_coords[id][0]], 'lon': [coord[1], true_coords[id][1]]}))
    else:
        st.write('No image uploaded.')
    
    
    # Footer
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align:center;">GeoGuessrAI App</h2>', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;font-size:12px;opacity:0.7;">Source code is <a href="https://github.com/yosshstd/GeoGuessrAI" target="_blank">here</a></div>',
        unsafe_allow_html=True
    )
    st.markdown('<br>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
