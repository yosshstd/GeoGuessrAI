SET_PAGE_CONFIG = {
    'page_title': 'GeoGuessrAI App',
    'page_icon': '🗺',
    'layout': 'centered', # 'centered', 'wide'
    'initial_sidebar_state': 'collapsed',
}

MODEL_URL = 'https://huggingface.co/yosshstd/GeoGuessrAI/resolve/main/model.pth'

DATA_URL = 'test_data/coord_images_test'

HIDE_ST_STYLE = '''
                <style>
                div[data-testid='stToolbar'] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid='stDecoration'] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
				        .appview-container .main .block-container{
                            padding-top: 1rem;
                            padding-right: 3rem;
                            padding-left: 3rem;
                            padding-bottom: 1rem;
                        }  
                        .reportview-container {
                            padding-top: 0rem;
                            padding-right: 3rem;
                            padding-left: 3rem;
                            padding-bottom: 0rem;
                        }
                        header[data-testid='stHeader'] {
                            z-index: -1;
                        }
                        div[data-testid='stToolbar'] {
                        z-index: 100;
                        }
                        div[data-testid='stDecoration'] {
                        z-index: 100;
                        }
                </style>
'''