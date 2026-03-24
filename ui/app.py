# ui/app.py
from __future__ import annotations

import os, sys, glob, time
import pandas as pd
import streamlit as st
from PIL import Image


# аргумент: --run_dir <path>
run_dir = 'runs/cifar10'
if '--' in sys.argv:
    idx = sys.argv.index('--')
    args = sys.argv[idx+1:]
    if len(args) >= 2 and args[0] == '--run_dir':
        run_dir = args[1]


st.set_page_config(page_title='4D-UV-Quat Diffusion', layout='wide')
st.title('4D-UV-Quat Diffusion — мониторинг')


log_path = os.path.join(run_dir, 'log.jsonl')
previews_dir = os.path.join(run_dir, 'previews')


col1, col2 = st.columns([2,1])


with col1:
    st.subheader('Графики обучения')
    if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
        df = pd.read_json(log_path, lines=True)
        df = df.sort_values('step')
        st.line_chart(df.set_index('step')[['loss','loss_eps','loss_uv']])
        st.caption(f"Последний шаг: {int(df['step'].max())}")
    else:
        st.info('Лог ещё не создан или пуст.')


with col2:
    st.subheader('Последний предпросмотр')
    if os.path.isdir(previews_dir):
        imgs = sorted(glob.glob(os.path.join(previews_dir, '*.png')), key=os.path.getmtime)
        if imgs:
            st.image(Image.open(imgs[-1]), use_column_width=True)
            st.caption(os.path.basename(imgs[-1]))
        else:
            st.info('Предпросмотров пока нет.')
    else:
        st.info('Папка предпросмотров не найдена.')


st.sidebar.header('Настройки')
st.sidebar.write(f'run_dir: {run_dir}')

# Автообновление через каждые 5 секунд
auto_refresh = st.sidebar.checkbox('Автообновление', value=True)
if auto_refresh:
    st.sidebar.info('Страница автообновляется каждые 5 секунд.')
    time.sleep(5)
    st.rerun()
else:
    st.sidebar.info('Автообновление отключено. Нажмите F5 для обновления.')

