import const
from src import lightning_model

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
import reverse_geocoder as rg
import pycountry

from st_img_pastebutton import paste
from io import BytesIO
import base64



def main():
    ''''''
    st.markdown(f'<h1 style="text-align:center;">GeoGuessrAI App</h1>', unsafe_allow_html=True)

    # Load the model cached
    @st.cache_resource
    def load_model():
        config = OmegaConf.load('config.yaml')
        model = lightning_model.BaseModel(config)
        state_dict = torch.hub.load_state_dict_from_url(const.MODEL_URL, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

        processor = CLIPImageProcessor.from_pretrained(config.data.model_name)
        return model, processor
    model, processor = load_model()
    

    # Load test image (jgp)
    @st.cache_resource
    def scan_image_path():
        paths = []
        true_coords = []
        for path in glob.iglob(const.DATA_URL+'/*.jpg'):
            paths.append(path)
            coord_list = path.split("/")[-1].split(",")
            lat, lon = [float(coord_list[0]), float(coord_list[1][:-4])]
            true_coords.append((lat, lon))
        
        true_results = rg.search(true_coords)
        return paths, true_coords, true_results
    paths, true_coords, true_results = scan_image_path()

    
    def get_country_name(country_code):
        try:
            country = pycountry.countries.get(alpha_2=country_code)
            return country.name
        except AttributeError:
            return "Unknown"
        
    def harversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    ''''''
    
    #col1, col2 = st.columns([1, 1])
    img_source = st.radio('Image Source', ('Sample', 'Paste', 'Upload'), help='You can paste a street view image from clipboard or upload an image from your local machine.')
    if img_source == 'Sample':
        try:
            id = np.random.randint(0, len(paths))
            image_data = Image.open(paths[id])
        except:
            image_data = None
    elif img_source == 'Paste':
        pasted_img = paste(key='image_clipboard', label='Paste an image from clipboard')
        try:
            header, encoded = pasted_img.split(",", 1)
            binary_data = base64.b64decode(encoded)
            image_data = Image.open(BytesIO(binary_data))
        except:
            image_data = None
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
                pred_coord = model(image_data).squeeze().cpu().numpy()
            
            pred_results = rg.search(pred_coord.tolist())
            if img_source == 'Sample':
                col1, col2 = st.columns([1, 2])
                col1.button('Next Street View Image', help='Click to see another sample image.')
                col2.success(f'Inference Time: {time.time()-start_time:.2f} [sec]')
            else:
                st.success(f'Inference Time: {time.time()-start_time:.2f} [sec]')

    else:
        st.info('Please upload an image.')
        pred_coord = None


    
    # Display the result
    st.subheader('Result')
    if pred_coord is not None:
        st.write(f'Predicted Location: ({pred_coord[0]:.2f}, {pred_coord[1]:.2f}) --- {pred_results[0]["name"]}, {pred_results[0]["admin1"]}, {get_country_name(pred_results[0]["cc"])}')
        if img_source == 'Sample':
            st.write(f'True Location: ({true_coords[id][0]:.2f}, {true_coords[id][1]:.2f}) --- {true_results[id]["name"]}, {true_results[id]["admin1"]}, {get_country_name(true_results[id]["cc"])}')
            st.map(pd.DataFrame({'lat': [pred_coord[0], true_coords[id][0]], 'lon': [pred_coord[1], true_coords[id][1]], 'type': ['Predicted', 'True'], 'color': ['#008000', '#0044ff']}), size=1000, zoom=1, color='color')
        else:
            st.map(pd.DataFrame({'lat': [pred_coord[0]], 'lon': [pred_coord[1]], 'type': ['Predicted'], 'color': ['#008000']}), size=1000, zoom=1, color='color')
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
