import sys, ast
sys.path.insert(0, '.')

for f in ['ui/styles.py', 'app.py']:
    with open(f, encoding='utf-8') as fh:
        ast.parse(fh.read())
    print(f'{f} sintaxis OK')

from ui.styles import (
    SOCIAL_ICONS, SOCIAL_MEDIA_DEFS,
    get_social_selector_html, get_model_badge_html, CUSTOM_CSS
)

keys = list(SOCIAL_ICONS.keys())
print(f'SOCIAL_ICONS cargados: {keys}')

all_uri = all(v.startswith('data:') for v in SOCIAL_ICONS.values())
print(f'Todos con data-URI: {all_uri}')

html = get_social_selector_html(['youtube', 'reddit'])
print(f'HTML chips len: {len(html)}')
selected_count = html.count('selected')
print(f'Chips selected count: {selected_count} (esperado 2)')
print(f'Contenedor OK: {"social-chips-container" in html}')

import gradio as gr
with gr.Blocks(title='Test') as demo:
    gr.HTML(html)
    st = gr.State(['youtube'])
    hid = gr.Textbox(value='youtube', visible=False, elem_id='social-hidden-input')
print('gr.Blocks con selector OK')
print('TODO OK')
