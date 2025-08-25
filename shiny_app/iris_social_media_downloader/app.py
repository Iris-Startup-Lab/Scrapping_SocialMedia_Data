# -*- coding: utf-8 -*-
## Iris Startup Lab 
'''
<(*)
  ( >)
  /|
'''

#-------------------------------------------------------------
######### Social Media Downloader Shiny App ######
######### VERSION 0.5 ######
######### Authors Fernando Dorantes Nieto
#-------------------------------------------------------------


# Este es el archivo principal que ejecuta la aplicación.
# Importa la interfaz de usuario (ui) y la lógica del servidor (server) desde sus respectivos módulos y los une para crear la aplicación Shiny.

import os 
from pathlib import Path
import shiny
from shiny import App

print("--- Environment Variable Check for Cache Config ---")
print(f"Initial os.environ.get('HF_SPACE_ID'): {os.environ.get('HF_SPACE_ID')}")
print(f"Initial os.environ.get('HOME'): {os.environ.get('HOME')}")
print(f"Initial os.environ.get('USER'): {os.environ.get('USER')}")
print(f"Initial os.environ.get('XDG_CACHE_HOME'): {os.environ.get('XDG_CACHE_HOME')}")
print("-------------------------------------------------")

try:
    # Para el servidor, detectando espacios de Hugging Face
    hf_space_id_value = os.environ.get('HF_SPACE_ID')
    is_huggingface_spaces_by_id = bool(hf_space_id_value)
    
    current_home_dir_str = os.path.expanduser('~') 
    current_home_dir = Path(current_home_dir_str)
    
    is_root_home = (current_home_dir_str == "/")

    print(f"DEBUG: HF_SPACE_ID raw value: '{hf_space_id_value}', is_huggingface_spaces_by_id: {is_huggingface_spaces_by_id}")
    print(f"DEBUG: os.path.expanduser('~') resolved to: {current_home_dir_str}, is_root_home: {is_root_home}")
    
    tmp_dir = Path("/tmp")
    tmp_exists = tmp_dir.exists()
    tmp_writable = os.access(str(tmp_dir), os.W_OK) if tmp_exists else False
    print(f"DEBUG: /tmp exists: {tmp_exists}, /tmp writable: {tmp_writable}")

    if is_huggingface_spaces_by_id:
        base_cache_path = tmp_dir / "iris_social_media_downloader_cache"
        print(f"INFO: Detected Hugging Face Spaces environment (by HF_SPACE_ID). Using /tmp for cache. Base path: {base_cache_path}")
    elif is_root_home and tmp_exists and tmp_writable:
        base_cache_path = tmp_dir / "iris_social_media_downloader_cache"
        print(f"INFO: Detected container-like environment (home is '/' and /tmp is writable). Using /tmp for cache. Base path: {base_cache_path}")
    else:
        can_write_to_home_cache = False
        if current_home_dir_str != "/":
            try:
                home_cache_test_path = current_home_dir / ".cache" / "_test_writability"
                os.makedirs(home_cache_test_path, exist_ok=True)
                os.rmdir(home_cache_test_path) 
                can_write_to_home_cache = True
            except OSError:
                can_write_to_home_cache = False
        
        if can_write_to_home_cache:
            base_cache_path = current_home_dir / ".cache" / "iris_social_media_downloader_cache"
            print(f"INFO: Detected standard local environment. Using home-based .cache: {base_cache_path}")
        else:
            script_dir_cache = Path(__file__).resolve().parent / ".app_cache" 
            base_cache_path = script_dir_cache / "iris_social_media_downloader_cache"
            print(f"INFO: Home dir ('{current_home_dir_str}') not suitable for .cache or /tmp fallback failed. Using script-relative cache: {base_cache_path}")

    os.makedirs(base_cache_path, exist_ok=True)
    print(f"DEBUG: Ensured base_cache_path exists: {base_cache_path}")
    
    hf_cache_path = base_cache_path / "huggingface"
    os.environ['HF_HOME'] = str(hf_cache_path)
    print(f"DEBUG: Setting HF_HOME to: {hf_cache_path}")

    mpl_cache_path = base_cache_path / "matplotlib"
    os.environ['MPLCONFIGDIR'] = str(mpl_cache_path)
    print(f"DEBUG: Setting MPLCONFIGDIR to: {mpl_cache_path}")


    os.environ['XDG_CACHE_HOME'] = str(base_cache_path)
    print(f"DEBUG: Setting XDG_CACHE_HOME to: {base_cache_path}")
    
    wdm_driver_cache = base_cache_path / "selenium" 
    os.environ['WDM_DRIVER_CACHE_PATH'] = str(wdm_driver_cache)
    print(f"DEBUG: Setting WDM_DRIVER_CACHE_PATH to: {wdm_driver_cache}")

    wdm_general_cache = base_cache_path / "webdriver_manager" 
    os.environ['WDM_LOCAL'] = str(wdm_general_cache)
    print(f"DEBUG: Setting WDM_LOCAL to: {wdm_general_cache}")

    os.makedirs(hf_cache_path, exist_ok=True)
    os.makedirs(mpl_cache_path, exist_ok=True)
    os.makedirs(wdm_driver_cache, exist_ok=True) 
    os.makedirs(wdm_general_cache, exist_ok=True)
    
    print(f"INFO: Final Cache directory base set to: {base_cache_path}")
    print(f"INFO: Final HF_HOME set to: {os.environ.get('HF_HOME')}")
    print(f"INFO: Final MPLCONFIGDIR set to: {os.environ.get('MPLCONFIGDIR')}")
    print(f"INFO: Final XDG_CACHE_HOME set to: {os.environ.get('XDG_CACHE_HOME')}")
    print(f"INFO: Final WDM_DRIVER_CACHE_PATH set to: {os.environ.get('WDM_DRIVER_CACHE_PATH')}")
    print(f"INFO: Final WDM_LOCAL set to: {os.environ.get('WDM_LOCAL')}")
except Exception as e: 
    print(f"CRITICAL WARNING: An unexpected error occurred during cache setup: {e}")
    import traceback
    traceback.print_exc()
    print("Proceeding without custom cache paths. This will likely lead to errors.")

# Importar los componentes desde el paquete app_module
from app_module.ui import app_ui
from app_module.server import server

# Opcional: Configurar el puerto si es necesario
# shiny.shiny_app.set_shiny_port(8000)

# Crear la instancia de la aplicación
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
