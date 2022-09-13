{ 
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "6b3f2578",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 모듈 가져오기\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from tqdm import tqdm\n",
    "import warnings\n",
    "from glob import glob\n",
    "from pandasql import sqldf\n",
    "import re\n",
    "import os\n",
    "import time\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras import layers\n",
    "\n",
    "pd.set_option('display.max_columns', None)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "d33a6b6b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 주산지 TOP3 날씨 정보를 평균내서 합치기\n",
    "def avg_weather_cols(df):\n",
    "    weather_elem = ['주산지_0_초기온도', '주산지_0_최대온도', '주산지_0_최저온도', '주산지_0_평균온도', '주산지_0_강수량', '주산지_0_습도']\n",
    "    \n",
    "    for elem in weather_elem:\n",
    "        result = re.sub('_0_', '_평균_', elem)\n",
    "        elem_0 = elem\n",
    "        elem_1 = re.sub('_0_', '_1_', elem)\n",
    "        elem_2 = re.sub('_0_', '_2_', elem)\n",
    "        \n",
    "        df = df.astype({elem_0:'float', elem_1:'float', elem_2:'float'})\n",
    "        df[result] = df[[elem_0, elem_1, elem_2]].mean(skipna=True, numeric_only=True, axis=1)\n",
    "        df[result] = df[result].interpolate()\n",
    "        \n",
    "        df.drop([elem_0, elem_1, elem_2], axis=1, inplace=True)\n",
    "    \n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "c8ed6169",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2주씩 train 시키기 위해 데이터 분할하기\n",
    "def time_window(df, t, t_sep):\n",
    "    seq_len = t\n",
    "    seqence_length = seq_len + t_sep\n",
    "\n",
    "    result = []\n",
    "    for index in tqdm(range(len(df) - seqence_length)):\n",
    "        result.append(df[index: index + seqence_length].values)\n",
    "\n",
    "    return np.array(result)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "f4265057",
   "metadata": {},
   "outputs": [],
   "source": [
    "data_list = glob(\"./aT_data/data/train/train*\")\n",
    "\n",
    "epoch = 10\n",
    "batch = 15\n",
    "learning_rate = 0.001\n",
    "\n",
    "output_dir = f'LSTM(relu)_{epoch}_{batch}'\n",
    "\n",
    "tr_del_list = ['단가', '거래량', '거래대금', '경매건수', '도매시장코드', '도매법인코드', '산지코드']  # train 에서 사용하지 않는 열\n",
    "ts_del_list = ['단가', '거래량', '거래대금', '경매건수', '도매시장코드', '도매법인코드', '산지코드', '해당일자_전체평균가격']  # test 에서 사용하지 않는 열\n",
    "check_col = ['일자구분_중순', '일자구분_초순', '일자구분_하순', '월구분_10월', '월구분_11월', '월구분_12월', '월구분_1월', '월구분_2월', '월구분_3월',\n",
    "             '월구분_4월', '월구분_5월', '월구분_6월', '월구분_7월', '월구분_8월', '월구분_9월']  # 열 개수 맞추기\n",
    "\n",
    "if os.path.exists(f'./results/{output_dir}') == False:\n",
    "    os.mkdir(f'./results/{output_dir}')\n",
    "\n",
    "if os.path.exists(f'./results/{output_dir}/models') == False:\n",
    "    os.mkdir(f'./results/{output_dir}/models')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "10866338",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  0%|                                                                                           | 0/37 [00:00<?, ?it/s]\n",
      "100%|███████████████████████████████████████████████████████████████████████████| 1433/1433 [00:00<00:00, 19396.52it/s]\u001b[A\n",
      "\n",
      "100%|███████████████████████████████████████████████████████████████████████████| 1405/1405 [00:00<00:00, 44414.78it/s]\u001b[A\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1124, 28, 37)\n",
      "(1124, 28)\n",
      "(281, 28, 37)\n",
      "(281, 28)\n",
      "Epoch 1/10\n",
      "75/75 [==============================] - 5s 23ms/step - loss: 0.0361 - val_loss: 0.0349\n",
      "Epoch 2/10\n",
      "75/75 [==============================] - 1s 17ms/step - loss: 0.0249 - val_loss: 0.0354\n",
      "Epoch 3/10\n",
      "75/75 [==============================] - 1s 16ms/step - loss: 0.0232 - val_loss: 0.0313\n",
      "Epoch 4/10\n",
      "75/75 [==============================] - 1s 16ms/step - loss: 0.0220 - val_loss: 0.0324\n",
      "Epoch 5/10\n",
      "75/75 [==============================] - 1s 16ms/step - loss: 0.0213 - val_loss: 0.0330\n",
      "Epoch 6/10\n",
      "75/75 [==============================] - 1s 16ms/step - loss: 0.0209 - val_loss: 0.0323\n",
      "Epoch 7/10\n",
      "75/75 [==============================] - 1s 17ms/step - loss: 0.0206 - val_loss: 0.0322\n",
      "Epoch 8/10\n",
      "75/75 [==============================] - 1s 17ms/step - loss: 0.0202 - val_loss: 0.0318\n",
      "Epoch 9/10\n",
      "75/75 [==============================] - 1s 17ms/step - loss: 0.0202 - val_loss: 0.0315\n",
      "Epoch 10/10\n",
      "75/75 [==============================] - 1s 17ms/step - loss: 0.0198 - val_loss: 0.0322\n",
      "9/9 [==============================] - 0s 7ms/step - loss: 0.0322\n",
      "9/9 [==============================] - 0s 7ms/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  3%|██▏                                                                                | 1/37 [00:17<10:13, 17.05s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R2 Score:  -12813475128.288\n",
      "MAE:  3686.325372299529\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "100%|███████████████████████████████████████████████████████████████████████████| 1433/1433 [00:00<00:00, 17542.17it/s]\u001b[A\n",
      "\n",
      "100%|███████████████████████████████████████████████████████████████████████████| 1405/1405 [00:00<00:00, 41036.73it/s]\u001b[A\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1124, 28, 37)\n",
      "(1124, 28)\n",
      "(281, 28, 37)\n",
      "(281, 28)\n",
      "Epoch 1/10\n",
      "75/75 [==============================] - 4s 25ms/step - loss: 0.0933 - val_loss: 0.0297\n",
      "Epoch 2/10\n",
      "75/75 [==============================] - 1s 19ms/step - loss: 0.0446 - val_loss: 0.0357\n",
      "Epoch 3/10\n",
      "75/75 [==============================] - 1s 17ms/step - loss: 0.0369 - val_loss: 0.0300\n",
      "Epoch 4/10\n",
      "75/75 [==============================] - 1s 18ms/step - loss: 0.0339 - val_loss: 0.0300\n",
      "Epoch 5/10\n",
      "75/75 [==============================] - 1s 17ms/step - loss: 0.0294 - val_loss: 0.0397\n",
      "Epoch 6/10\n",
      "75/75 [==============================] - 1s 17ms/step - loss: 0.0283 - val_loss: 0.0342\n",
      "Epoch 7/10\n",
      "75/75 [==============================] - 1s 16ms/step - loss: 0.0264 - val_loss: 0.0359\n",
      "Epoch 8/10\n",
      "75/75 [==============================] - 1s 17ms/step - loss: 0.0243 - val_loss: 0.0370\n",
      "Epoch 9/10\n",
      "75/75 [==============================] - 1s 17ms/step - loss: 0.0247 - val_loss: 0.0306\n",
      "Epoch 10/10\n",
      "75/75 [==============================] - 1s 17ms/step - loss: 0.0237 - val_loss: 0.0346\n",
      "9/9 [==============================] - 0s 5ms/step - loss: 0.0346\n",
      "9/9 [==============================] - 0s 7ms/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  5%|████▍                                                                              | 2/37 [00:34<09:58, 17.10s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R2 Score:  -576240021.1183\n",
      "MAE:  1165.2686406234877\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "100%|███████████████████████████████████████████████████████████████████████████| 1433/1433 [00:00<00:00, 17733.98it/s]\u001b[A\n",
      "\n",
      "100%|███████████████████████████████████████████████████████████████████████████| 1405/1405 [00:00<00:00, 46993.97it/s]\u001b[A\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1124, 28, 37)\n",
      "(1124, 28)\n",
      "(281, 28, 37)\n",
      "(281, 28)\n",
      "Epoch 1/10\n",
      "75/75 [==============================] - 4s 25ms/step - loss: 0.0866 - val_loss: 0.0523\n",
      "Epoch 2/10\n",
      "75/75 [==============================] - 1s 18ms/step - loss: 0.0453 - val_loss: 0.0503\n",
      "Epoch 3/10\n",
      "75/75 [==============================] - 1s 17ms/step - loss: 0.0372 - val_loss: 0.0484\n",
      "Epoch 4/10\n",
      "75/75 [==============================] - 1s 17ms/step - loss: 0.0328 - val_loss: 0.0473\n",
      "Epoch 5/10\n",
      "75/75 [==============================] - 1s 17ms/step - loss: 0.0309 - val_loss: 0.0464\n",
      "Epoch 6/10\n",
      "75/75 [==============================] - 1s 17ms/step - loss: 0.0290 - val_loss: 0.0468\n",
      "Epoch 7/10\n",
      "75/75 [==============================] - 1s 17ms/step - loss: 0.0281 - val_loss: 0.0453\n",
      "Epoch 8/10\n",
      "75/75 [==============================] - 1s 18ms/step - loss: 0.0272 - val_loss: 0.0473\n",
      "Epoch 9/10\n",
      "75/75 [==============================] - 1s 18ms/step - loss: 0.0269 - val_loss: 0.0465\n",
      "Epoch 10/10\n",
      "75/75 [==============================] - 1s 17ms/step - loss: 0.0260 - val_loss: 0.0446\n",
      "9/9 [==============================] - 0s 8ms/step - loss: 0.0446\n",
      "9/9 [==============================] - 0s 8ms/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  8%|██████▋                                                                            | 3/37 [00:51<09:43, 17.16s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R2 Score:  -709772477.7993157\n",
      "MAE:  1731.6723422892126\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "100%|███████████████████████████████████████████████████████████████████████████| 1433/1433 [00:00<00:00, 17229.83it/s]\u001b[A\n",
      "\n",
      "100%|███████████████████████████████████████████████████████████████████████████| 1405/1405 [00:00<00:00, 46460.45it/s]\u001b[A\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1124, 28, 37)\n",
      "(1124, 28)\n",
      "(281, 28, 37)\n",
      "(281, 28)\n",
      "Epoch 1/10\n",
      "75/75 [==============================] - 4s 23ms/step - loss: 0.1743 - val_loss: 0.1271\n",
      "Epoch 2/10\n",
      "75/75 [==============================] - 1s 17ms/step - loss: 0.0875 - val_loss: 0.1100\n",
      "Epoch 3/10\n",
      "75/75 [==============================] - 1s 17ms/step - loss: 0.0778 - val_loss: 0.1181\n",
      "Epoch 4/10\n",
      "75/75 [==============================] - 1s 18ms/step - loss: 0.0633 - val_loss: 0.0892\n",
      "Epoch 5/10\n",
      "75/75 [==============================] - 1s 19ms/step - loss: 0.0587 - val_loss: 0.0915\n",
      "Epoch 6/10\n",
      "75/75 [==============================] - 1s 19ms/step - loss: 0.0554 - val_loss: 0.1013\n",
      "Epoch 7/10\n",
      "75/75 [==============================] - 1s 18ms/step - loss: 0.0541 - val_loss: 0.1087\n",
      "Epoch 8/10\n",
      "75/75 [==============================] - 1s 17ms/step - loss: 0.0490 - val_loss: 0.1054\n",
      "Epoch 9/10\n",
      "75/75 [==============================] - 1s 18ms/step - loss: 0.0472 - val_loss: 0.1006\n",
      "Epoch 10/10\n",
      "75/75 [==============================] - 1s 18ms/step - loss: 0.0473 - val_loss: 0.1112\n",
      "9/9 [==============================] - 0s 8ms/step - loss: 0.1112\n",
      "9/9 [==============================] - 0s 8ms/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 11%|████████▉                                                                          | 4/37 [01:09<09:35, 17.45s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R2 Score:  -226025098.83129266\n",
      "MAE:  4015.2742680281435\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "100%|███████████████████████████████████████████████████████████████████████████| 1433/1433 [00:00<00:00, 19328.53it/s]\u001b[A\n",
      "\n",
      "100%|███████████████████████████████████████████████████████████████████████████| 1405/1405 [00:00<00:00, 50224.12it/s]\u001b[A\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1124, 28, 37)\n",
      "(1124, 28)\n",
      "(281, 28, 37)\n",
      "(281, 28)\n",
      "Epoch 1/10\n",
      "75/75 [==============================] - 5s 26ms/step - loss: 0.1769 - val_loss: 0.2204\n",
      "Epoch 2/10\n",
      "75/75 [==============================] - 1s 18ms/step - loss: 0.1043 - val_loss: 0.1648\n",
      "Epoch 3/10\n",
      "75/75 [==============================] - 1s 18ms/step - loss: 0.0898 - val_loss: 0.2000\n",
      "Epoch 4/10\n",
      "75/75 [==============================] - 1s 18ms/step - loss: 0.0787 - val_loss: 0.1562\n",
      "Epoch 5/10\n",
      "75/75 [==============================] - 1s 18ms/step - loss: 0.0731 - val_loss: 0.1382\n",
      "Epoch 6/10\n",
      "75/75 [==============================] - 1s 19ms/step - loss: 0.0698 - val_loss: 0.1571\n",
      "Epoch 7/10\n",
      "75/75 [==============================] - 2s 20ms/step - loss: 0.0664 - val_loss: 0.1603\n",
      "Epoch 8/10\n",
      "75/75 [==============================] - 2s 20ms/step - loss: 0.0661 - val_loss: 0.1765\n",
      "Epoch 9/10\n",
      "75/75 [==============================] - 2s 21ms/step - loss: 0.0643 - val_loss: 0.1573\n",
      "Epoch 10/10\n",
      "75/75 [==============================] - 2s 21ms/step - loss: 0.0633 - val_loss: 0.1802\n",
      "9/9 [==============================] - 0s 9ms/step - loss: 0.1802\n",
      "9/9 [==============================] - 0s 5ms/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 14%|███████████▏                                                                       | 5/37 [01:28<09:35, 17.98s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R2 Score:  -619125104.8153974\n",
      "MAE:  3052.529560635761\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "100%|███████████████████████████████████████████████████████████████████████████| 1433/1433 [00:00<00:00, 23372.64it/s]\u001b[A\n",
      "\n",
      "100%|███████████████████████████████████████████████████████████████████████████| 1405/1405 [00:00<00:00, 48357.16it/s]\u001b[A\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1124, 28, 37)\n",
      "(1124, 28)\n",
      "(281, 28, 37)\n",
      "(281, 28)\n",
      "Epoch 1/10\n",
      "75/75 [==============================] - 4s 26ms/step - loss: 0.1434 - val_loss: 0.0879\n",
      "Epoch 2/10\n",
      "75/75 [==============================] - 1s 18ms/step - loss: 0.0772 - val_loss: 0.0906\n",
      "Epoch 3/10\n",
      "75/75 [==============================] - 1s 19ms/step - loss: 0.0635 - val_loss: 0.1000\n",
      "Epoch 4/10\n",
      "75/75 [==============================] - 1s 18ms/step - loss: 0.0568 - val_loss: 0.0885\n",
      "Epoch 5/10\n",
      "75/75 [==============================] - 1s 19ms/step - loss: 0.0531 - val_loss: 0.1120\n",
      "Epoch 6/10\n",
      "75/75 [==============================] - 2s 20ms/step - loss: 0.0486 - val_loss: 0.0985\n",
      "Epoch 7/10\n",
      "75/75 [==============================] - 2s 20ms/step - loss: 0.0456 - val_loss: 0.1027\n",
      "Epoch 8/10\n",
      "75/75 [==============================] - 2s 20ms/step - loss: 0.0459 - val_loss: 0.0878\n",
      "Epoch 9/10\n",
      "75/75 [==============================] - 1s 18ms/step - loss: 0.0437 - val_loss: 0.1094\n",
      "Epoch 10/10\n",
      "75/75 [==============================] - 1s 19ms/step - loss: 0.0424 - val_loss: 0.1029\n",
      "9/9 [==============================] - 0s 7ms/step - loss: 0.1029\n",
      "9/9 [==============================] - 0s 8ms/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 16%|█████████████▍                                                                     | 6/37 [01:46<09:23, 18.17s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "R2 Score:  -691364505.38332\n",
      "MAE:  2178.964411021388\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "100%|███████████████████████████████████████████████████████████████████████████| 1433/1433 [00:00<00:00, 21742.92it/s]\u001b[A\n",
      "\n",
      "100%|███████████████████████████████████████████████████████████████████████████| 1405/1405 [00:00<00:00, 46996.22it/s]\u001b[A\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:461: RuntimeWarning: All-NaN slice encountered\n",
      "  data_min = np.nanmin(X, axis=0)\n",
      "C:\\Users\\Playdata\\anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\_data.py:462: RuntimeWarning: All-NaN slice encountered\n",
      "  data_max = np.nanmax(X, axis=0)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1124, 28, 37)\n",
      "(1124, 28)\n",
      "(281, 28, 37)\n",
      "(281, 28)\n",
      "Epoch 1/10\n",
      "75/75 [==============================] - 5s 27ms/step - loss: nan - val_loss: nan\n",
      "Epoch 2/10\n",
      "75/75 [==============================] - 2s 20ms/step - loss: nan - val_loss: nan\n",
      "Epoch 3/10\n",
      "75/75 [==============================] - 1s 20ms/step - loss: nan - val_loss: nan\n",
      "Epoch 4/10\n",
      "75/75 [==============================] - 2s 20ms/step - loss: nan - val_loss: nan\n",
      "Epoch 5/10\n",
      "75/75 [==============================] - 2s 20ms/step - loss: nan - val_loss: nan\n",
      "Epoch 6/10\n",
      "75/75 [==============================] - 2s 20ms/step - loss: nan - val_loss: nan\n",
      "Epoch 7/10\n",
      "75/75 [==============================] - 2s 20ms/step - loss: nan - val_loss: nan\n",
      "Epoch 8/10\n",
      "75/75 [==============================] - 2s 20ms/step - loss: nan - val_loss: nan\n",
      "Epoch 9/10\n",
      "75/75 [==============================] - 2s 20ms/step - loss: nan - val_loss: nan\n",
      "Epoch 10/10\n",
      "75/75 [==============================] - 1s 20ms/step - loss: nan - val_loss: nan\n",
      "9/9 [==============================] - 0s 9ms/step - loss: nan\n",
      "9/9 [==============================] - 0s 9ms/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 16%|█████████████▍                                                                     | 6/37 [02:07<11:00, 21.31s/it]\n"
     ]
    },
    {
     "ename": "ValueError",
     "evalue": "Input contains NaN, infinity or a value too large for dtype('float32').",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Input \u001b[1;32mIn [11]\u001b[0m, in \u001b[0;36m<cell line: 3>\u001b[1;34m()\u001b[0m\n\u001b[0;32m    133\u001b[0m     y_pred_inverse \u001b[38;5;241m=\u001b[39m y_scaler\u001b[38;5;241m.\u001b[39minverse_transform(y_pred)\n\u001b[0;32m    134\u001b[0m \u001b[38;5;66;03m#     print(y_pred_inverse.shape)\u001b[39;00m\n\u001b[1;32m--> 136\u001b[0m     r2_result \u001b[38;5;241m=\u001b[39m \u001b[43mr2_score\u001b[49m\u001b[43m(\u001b[49m\u001b[43my_val\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43my_pred_inverse\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    137\u001b[0m     mae_result \u001b[38;5;241m=\u001b[39m mean_absolute_error(y_val, y_pred_inverse)\n\u001b[0;32m    139\u001b[0m     \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mR2 Score: \u001b[39m\u001b[38;5;124m'\u001b[39m, r2_result)\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\metrics\\_regression.py:789\u001b[0m, in \u001b[0;36mr2_score\u001b[1;34m(y_true, y_pred, sample_weight, multioutput)\u001b[0m\n\u001b[0;32m    702\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mr2_score\u001b[39m(y_true, y_pred, \u001b[38;5;241m*\u001b[39m, sample_weight\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m, multioutput\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124muniform_average\u001b[39m\u001b[38;5;124m\"\u001b[39m):\n\u001b[0;32m    703\u001b[0m     \u001b[38;5;124;03m\"\"\":math:`R^2` (coefficient of determination) regression score function.\u001b[39;00m\n\u001b[0;32m    704\u001b[0m \n\u001b[0;32m    705\u001b[0m \u001b[38;5;124;03m    Best possible score is 1.0 and it can be negative (because the\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    787\u001b[0m \u001b[38;5;124;03m    -3.0\u001b[39;00m\n\u001b[0;32m    788\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[1;32m--> 789\u001b[0m     y_type, y_true, y_pred, multioutput \u001b[38;5;241m=\u001b[39m \u001b[43m_check_reg_targets\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m    790\u001b[0m \u001b[43m        \u001b[49m\u001b[43my_true\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43my_pred\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mmultioutput\u001b[49m\n\u001b[0;32m    791\u001b[0m \u001b[43m    \u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    792\u001b[0m     check_consistent_length(y_true, y_pred, sample_weight)\n\u001b[0;32m    794\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m _num_samples(y_pred) \u001b[38;5;241m<\u001b[39m \u001b[38;5;241m2\u001b[39m:\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\metrics\\_regression.py:96\u001b[0m, in \u001b[0;36m_check_reg_targets\u001b[1;34m(y_true, y_pred, multioutput, dtype)\u001b[0m\n\u001b[0;32m     94\u001b[0m check_consistent_length(y_true, y_pred)\n\u001b[0;32m     95\u001b[0m y_true \u001b[38;5;241m=\u001b[39m check_array(y_true, ensure_2d\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mFalse\u001b[39;00m, dtype\u001b[38;5;241m=\u001b[39mdtype)\n\u001b[1;32m---> 96\u001b[0m y_pred \u001b[38;5;241m=\u001b[39m \u001b[43mcheck_array\u001b[49m\u001b[43m(\u001b[49m\u001b[43my_pred\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mensure_2d\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mFalse\u001b[39;49;00m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mdtype\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mdtype\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m     98\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m y_true\u001b[38;5;241m.\u001b[39mndim \u001b[38;5;241m==\u001b[39m \u001b[38;5;241m1\u001b[39m:\n\u001b[0;32m     99\u001b[0m     y_true \u001b[38;5;241m=\u001b[39m y_true\u001b[38;5;241m.\u001b[39mreshape((\u001b[38;5;241m-\u001b[39m\u001b[38;5;241m1\u001b[39m, \u001b[38;5;241m1\u001b[39m))\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:800\u001b[0m, in \u001b[0;36mcheck_array\u001b[1;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)\u001b[0m\n\u001b[0;32m    794\u001b[0m         \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[0;32m    795\u001b[0m             \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mFound array with dim \u001b[39m\u001b[38;5;132;01m%d\u001b[39;00m\u001b[38;5;124m. \u001b[39m\u001b[38;5;132;01m%s\u001b[39;00m\u001b[38;5;124m expected <= 2.\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    796\u001b[0m             \u001b[38;5;241m%\u001b[39m (array\u001b[38;5;241m.\u001b[39mndim, estimator_name)\n\u001b[0;32m    797\u001b[0m         )\n\u001b[0;32m    799\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m force_all_finite:\n\u001b[1;32m--> 800\u001b[0m         \u001b[43m_assert_all_finite\u001b[49m\u001b[43m(\u001b[49m\u001b[43marray\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mallow_nan\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mforce_all_finite\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m==\u001b[39;49m\u001b[43m \u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mallow-nan\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m)\u001b[49m\n\u001b[0;32m    802\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m ensure_min_samples \u001b[38;5;241m>\u001b[39m \u001b[38;5;241m0\u001b[39m:\n\u001b[0;32m    803\u001b[0m     n_samples \u001b[38;5;241m=\u001b[39m _num_samples(array)\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:114\u001b[0m, in \u001b[0;36m_assert_all_finite\u001b[1;34m(X, allow_nan, msg_dtype)\u001b[0m\n\u001b[0;32m    107\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m (\n\u001b[0;32m    108\u001b[0m         allow_nan\n\u001b[0;32m    109\u001b[0m         \u001b[38;5;129;01mand\u001b[39;00m np\u001b[38;5;241m.\u001b[39misinf(X)\u001b[38;5;241m.\u001b[39many()\n\u001b[0;32m    110\u001b[0m         \u001b[38;5;129;01mor\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m allow_nan\n\u001b[0;32m    111\u001b[0m         \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m np\u001b[38;5;241m.\u001b[39misfinite(X)\u001b[38;5;241m.\u001b[39mall()\n\u001b[0;32m    112\u001b[0m     ):\n\u001b[0;32m    113\u001b[0m         type_err \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124minfinity\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;28;01mif\u001b[39;00m allow_nan \u001b[38;5;28;01melse\u001b[39;00m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mNaN, infinity\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m--> 114\u001b[0m         \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[0;32m    115\u001b[0m             msg_err\u001b[38;5;241m.\u001b[39mformat(\n\u001b[0;32m    116\u001b[0m                 type_err, msg_dtype \u001b[38;5;28;01mif\u001b[39;00m msg_dtype \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m \u001b[38;5;28;01melse\u001b[39;00m X\u001b[38;5;241m.\u001b[39mdtype\n\u001b[0;32m    117\u001b[0m             )\n\u001b[0;32m    118\u001b[0m         )\n\u001b[0;32m    119\u001b[0m \u001b[38;5;66;03m# for object dtype data, we only check for NaNs (GH-13254)\u001b[39;00m\n\u001b[0;32m    120\u001b[0m \u001b[38;5;28;01melif\u001b[39;00m X\u001b[38;5;241m.\u001b[39mdtype \u001b[38;5;241m==\u001b[39m np\u001b[38;5;241m.\u001b[39mdtype(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mobject\u001b[39m\u001b[38;5;124m\"\u001b[39m) \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m allow_nan:\n",
      "\u001b[1;31mValueError\u001b[0m: Input contains NaN, infinity or a value too large for dtype('float32')."
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 432x288 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "st = time.time()\n",
    "\n",
    "for data in tqdm(data_list):\n",
    "    df_number = data.split(\"_\")[-1].split(\".\")[0]\n",
    "    df = pd.read_csv(data).interpolate()\n",
    "\n",
    "    new_cols = []\n",
    "    for col in df.columns:\n",
    "        df[col] = df[col].replace({' ': np.nan})\n",
    "        new_cols.append(re.sub('_$', '', re.sub('\\(.*\\)', '', col.replace(' ', '_'))))\n",
    "\n",
    "    # 칼럼명 수정\n",
    "    df.columns = new_cols\n",
    "#     print(df.columns)\n",
    "\n",
    "    # 사용할 열 선택 및 index 설정\n",
    "    df.drop(tr_del_list, axis=1, inplace=True)\n",
    "    df.set_index('datadate', drop=True, inplace=True)\n",
    "    \n",
    "    # 날씨 정보 평균으로 계산하기\n",
    "    df = avg_weather_cols(df)\n",
    "\n",
    "    # 사용할 열 선택 및 index 설정\n",
    "    df['해당일자_전체평균가격'].fillna(df['해당일자_전체평균가격'].mean(), inplace=True)\n",
    "    df['해당일자_전체거래물량'].fillna(df['해당일자_전체거래물량'].mean(), inplace=True)\n",
    "    df['하위가격_평균가'].fillna(df['하위가격_평균가'].mean(), inplace=True)\n",
    "    df['상위가격_평균가'].fillna(df['상위가격_평균가'].mean(), inplace=True)\n",
    "    df['하위가격_거래물량'].fillna(df['하위가격_거래물량'].mean(), inplace=True)\n",
    "    df['상위가격_거래물량'].fillna(df['상위가격_거래물량'].mean(), inplace=True)\n",
    "    df['일자별_도매가격_최대'].fillna(df['일자별_도매가격_최대'].mean(), inplace=True)\n",
    "    df['일자별_도매가격_평균'].fillna(df['일자별_도매가격_평균'].mean(), inplace=True)\n",
    "    df['일자별_도매가격_최소'].fillna(df['일자별_도매가격_최소'].mean(), inplace=True)\n",
    "    df['일자별_소매가격_최대'].fillna(df['일자별_소매가격_최대'].mean(), inplace=True)\n",
    "    df['일자별_소매가격_평균'].fillna(df['일자별_소매가격_평균'].mean(), inplace=True)\n",
    "    df['일자별_소매가격_최소'].fillna(df['일자별_소매가격_최소'].mean(), inplace=True)\n",
    "  \n",
    "\n",
    "#     # nan 처리\n",
    "#     df = df.fillna(0)\n",
    "    \n",
    "    # 변수와 타겟 분리\n",
    "    x, y = df[[col for col in df.columns if col != '해당일자_전체평균가격']], df['해당일자_전체평균가격']\n",
    "\n",
    "    # 2주 입력을 통한 이후 4주 예측을 위해 y의 첫 14일을 제외\n",
    "    y = y[28:]\n",
    "\n",
    "    # time series window 생성\n",
    "    data_x = time_window(x, 27, 1)\n",
    "    data_y = time_window(y, 27, 1)\n",
    "\n",
    "    # y의 길이와 같은 길이로 설정\n",
    "    xdata = data_x[:len(data_y)]\n",
    "    ydata = data_y\n",
    "    \n",
    "    # train, validation 분리 (8 : 2)\n",
    "    x_train, x_val, y_train, y_val = train_test_split(xdata, ydata, test_size=0.2, shuffle=False, random_state=119)\n",
    "\n",
    "    # 데이터 정규화\n",
    "    x_scaler = MinMaxScaler()\n",
    "    y_scaler = MinMaxScaler()\n",
    "    \n",
    "    for i in range(x_train.shape[1]):\n",
    "        x_scaler.partial_fit(x_train[:, i, :])\n",
    "    results = []\n",
    "    for i in range(x_train.shape[1]):\n",
    "        results.append(x_scaler.transform(x_train[:, i, :]).reshape(x_train.shape[0], 1, x_train.shape[2]))\n",
    "    x_train = np.concatenate(results, axis=1)\n",
    "    \n",
    "    results = []\n",
    "    for i in range(x_val.shape[1]):\n",
    "        results.append(x_scaler.transform(x_val[:, i, :]).reshape(x_val.shape[0], 1, x_val.shape[2]))\n",
    "    x_val = np.concatenate(results, axis=1)    \n",
    "    \n",
    "    \n",
    "    y_scaler = y_scaler.fit(y_train)\n",
    "    y_train = y_scaler.transform(y_train)\n",
    "    y_val = y_scaler.transform(y_val)\n",
    "    \n",
    "\n",
    "#     df.columns = ['해당일자_전체평균가격', '해당일자_전체거래물량', '하위가격_평균가', '상위가격_평균가', '하위가격_거래물량', \n",
    "#                   '상위가격_거래물량', '일자별_도매가격_최대', '일자별_도매가격_평균', '일자별_도매가격_최소', '일자별_소매가격_최대',\n",
    "#                   '일자별_소매가격_평균', '일자별_소매가격_최소', '수출중량', '수출금액', '수입중량', '수입금액', '무역수지',\n",
    "#                   '일자구분_중순', '일자구분_초순', '일자구분_하순', '월구분_10월', '월구분_11월', '월구분_12월', '월구분_1월', \n",
    "#                   '월구분_2월', '월구분_3월', '월구분_4월', '월구분_5월', '월구분_6월', '월구분_7월', '월구분_8월', '월구분_9월', \n",
    "#                   '주산지_평균_초기온도', '주산지_평균_최대온도', '주산지_평균_최저온도', '주산지_평균_평균온도', '주산지_평균_강수량', \n",
    "#                   '주산지_평균_습도']\n",
    "\n",
    "    print(x_train.shape)\n",
    "    print(y_train.shape)\n",
    "    print(x_val.shape)\n",
    "    print(y_val.shape)\n",
    "    \n",
    "    # transformer 모델 훈련\n",
    "    model = keras.Sequential([\n",
    "        layers.LSTM(units=50, \n",
    "                    return_sequences=True, \n",
    "                    input_shape = x_train.shape[1:],\n",
    "                    activation='relu'),\n",
    "        layers.Dropout(0.1),\n",
    "        layers.LSTM(units=50, activation='relu', return_sequences=True),\n",
    "        layers.Dropout(0.1),\n",
    "        layers.LSTM(units=50, activation='relu'),\n",
    "        layers.Dropout(0.1),\n",
    "        layers.Dense(28)\n",
    "    ])\n",
    "\n",
    "    model.compile(optimizer='adam', loss='mae')\n",
    "\n",
    "    \n",
    "    # LSTM 네트워크 학습\n",
    "    history = model.fit(x_train, y_train, epochs=epoch, batch_size=batch, validation_data=(x_val, y_val), callbacks=EarlyStopping(patience=30))\n",
    "    \n",
    "    # 학습 결과 확인\n",
    "    plt.plot(history.history['loss'], 'b-', label='loss')\n",
    "    plt.plot(history.history['val_loss'], 'r--', label='val_loss')\n",
    "    plt.xlabel('Epoch')\n",
    "    plt.legend()\n",
    "    plt.savefig(f'./results/{output_dir}/model_history_{df_number}.png', bbox_inches='tight')\n",
    "    plt.clf()\n",
    "    \n",
    "    # 모델 저장\n",
    "    model.save(f'./results/{output_dir}/models/model_{df_number}.h5')\n",
    "    model.save_weights(f'./results/{output_dir}/models/model_weights_{df_number}.h5')\n",
    "    \n",
    "    # Test 데이터에 대한 예측 정확도 확인\n",
    "    evaluation = model.evaluate(x_val, y_val)\n",
    "\n",
    "    y_pred = model.predict(x_val)\n",
    "    y_pred_inverse = y_scaler.inverse_transform(y_pred)   # invert the prediction to understandable values\n",
    "    \n",
    "    r2_result = r2_score(y_val, y_pred_inverse)\n",
    "    mae_result = mean_absolute_error(y_val, y_pred_inverse)\n",
    "    \n",
    "    print('R2 Score: ', r2_result)\n",
    "    print('MAE: ', mae_result)\n",
    "    \n",
    "    # 모델 관련 내용 저장\n",
    "    with open(f'./results/{output_dir}/model_summary_{df_number}.txt', 'w') as f:\n",
    "        f.write(f'epoch = {epoch}\\n')\n",
    "        f.write(f'batch = {batch}\\n')\n",
    "        f.write(f'learning_rate = {learning_rate}\\n')\n",
    "        f.write('\\n***** LSTM Model Summary *****\\n')\n",
    "        model.summary(print_fn=lambda x: f.write(x + '\\n'))\n",
    "        f.write('\\n***** Prediction Results *****\\n')\n",
    "        f.write(f'Evaluation: {evaluation}\\n')\n",
    "        f.write(f'R2 Score: {r2_result}\\n')\n",
    "        f.write(f'MAE: {mae_result}\\n')\n",
    "    \n",
    "et = time.time()\n",
    "total_time = (et-st)/60\n",
    "print(f'{total_time} minutes')\n",
    "\n",
    "with open(f'./results/{output_dir}/README.txt', 'w') as f:\n",
    "    f.write(f'{total_time} minutes')\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a0851d2b",
   "metadata": {},
   "outputs": [],
   "source": [
    "for data in tqdm(data_list):\n",
    "    df_number = data.split(\"_\")[-1].split(\".\")[0]\n",
    "    \n",
    "    model.load(f'./results/{output_dir}/models/model_{df_number}.h5')\n",
    "#     model.save_weights(f'./results/{output_dir}/models/model_weights_{df_number}.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "b34b53ef",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  3%|██▏                                                                                | 1/37 [00:00<00:06,  5.48it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0\n",
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n",
      "1\n",
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  8%|██████▋                                                                            | 3/37 [00:00<00:06,  5.49it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2\n",
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n",
      "3\n",
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 14%|███████████▏                                                                       | 5/37 [00:00<00:05,  5.94it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "4\n",
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n",
      "5\n",
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 19%|███████████████▋                                                                   | 7/37 [00:01<00:06,  4.93it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "6\n",
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 22%|█████████████████▉                                                                 | 8/37 [00:01<00:06,  4.60it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "7\n",
      "             count\n",
      "수출중량          1461\n",
      "수출금액          1461\n",
      "수입중량          1461\n",
      "수입금액          1461\n",
      "무역수지          1461\n",
      "주산지_평균_초기온도     33\n",
      "주산지_평균_최대온도     33\n",
      "주산지_평균_최저온도     33\n",
      "주산지_평균_평균온도     33\n",
      "주산지_평균_강수량      33\n",
      "주산지_평균_습도     1063\n",
      "\n",
      "8\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 24%|████████████████████▏                                                              | 9/37 [00:01<00:05,  4.85it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "      count\n",
      "수출중량   1461\n",
      "수출금액   1461\n",
      "수입중량   1461\n",
      "수입금액   1461\n",
      "무역수지   1461\n",
      "\n",
      "9\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 27%|██████████████████████▏                                                           | 10/37 [00:01<00:05,  4.92it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 30%|████████████████████████▍                                                         | 11/37 [00:02<00:05,  4.53it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "10\n",
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n",
      "11\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 32%|██████████████████████████▌                                                       | 12/37 [00:02<00:05,  4.67it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n",
      "12\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 35%|████████████████████████████▊                                                     | 13/37 [00:02<00:04,  4.84it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 38%|███████████████████████████████                                                   | 14/37 [00:02<00:04,  4.71it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "13\n",
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 41%|█████████████████████████████████▏                                                | 15/37 [00:03<00:04,  4.55it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "14\n",
      "           count\n",
      "수출중량        1461\n",
      "수출금액        1461\n",
      "수입중량        1461\n",
      "수입금액        1461\n",
      "무역수지        1461\n",
      "주산지_평균_습도    136\n",
      "\n",
      "15\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 43%|███████████████████████████████████▍                                              | 16/37 [00:03<00:04,  4.56it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n",
      "16\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 46%|█████████████████████████████████████▋                                            | 17/37 [00:03<00:04,  4.82it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "      count\n",
      "수출중량   1461\n",
      "수출금액   1461\n",
      "수입중량   1461\n",
      "수입금액   1461\n",
      "무역수지   1461\n",
      "\n",
      "17\n",
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 51%|██████████████████████████████████████████                                        | 19/37 [00:03<00:03,  5.14it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "18\n",
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 54%|████████████████████████████████████████████▎                                     | 20/37 [00:04<00:03,  4.40it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "19\n",
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n",
      "20\n",
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 59%|████████████████████████████████████████████████▊                                 | 22/37 [00:04<00:03,  4.53it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "21\n",
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n",
      "22\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 62%|██████████████████████████████████████████████████▉                               | 23/37 [00:04<00:02,  4.67it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n",
      "23\n",
      "      count\n",
      "수출중량   1461\n",
      "수출금액   1461\n",
      "수입중량   1461\n",
      "수입금액   1461\n",
      "무역수지   1461\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 68%|███████████████████████████████████████████████████████▍                          | 25/37 [00:05<00:02,  4.67it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "24\n",
      "      count\n",
      "수출중량   1461\n",
      "수출금액   1461\n",
      "수입중량   1461\n",
      "수입금액   1461\n",
      "무역수지   1461\n",
      "\n",
      "25\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 70%|█████████████████████████████████████████████████████████▌                        | 26/37 [00:05<00:02,  4.87it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "           count\n",
      "주산지_평균_습도    136\n",
      "\n",
      "26\n",
      "      count\n",
      "수출중량   1461\n",
      "수출금액   1461\n",
      "수입중량   1461\n",
      "수입금액   1461\n",
      "무역수지   1461\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 76%|██████████████████████████████████████████████████████████████                    | 28/37 [00:05<00:01,  4.96it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "27\n",
      "      count\n",
      "수출중량   1461\n",
      "수출금액   1461\n",
      "수입중량   1461\n",
      "수입금액   1461\n",
      "무역수지   1461\n",
      "\n",
      "28\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 78%|████████████████████████████████████████████████████████████████▎                 | 29/37 [00:05<00:01,  5.05it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n",
      "29\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 81%|██████████████████████████████████████████████████████████████████▍               | 30/37 [00:06<00:01,  5.14it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 84%|████████████████████████████████████████████████████████████████████▋             | 31/37 [00:06<00:01,  4.35it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "30\n",
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n",
      "31\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 86%|██████████████████████████████████████████████████████████████████████▉           | 32/37 [00:06<00:01,  4.60it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "      count\n",
      "수출중량   1461\n",
      "수출금액   1461\n",
      "수입중량   1461\n",
      "수입금액   1461\n",
      "무역수지   1461\n",
      "\n",
      "32\n",
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 92%|███████████████████████████████████████████████████████████████████████████▎      | 34/37 [00:06<00:00,  4.93it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "33\n",
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n",
      "34\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 95%|█████████████████████████████████████████████████████████████████████████████▌    | 35/37 [00:07<00:00,  5.13it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n",
      "35\n",
      "           count\n",
      "주산지_평균_습도    136\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████████████████████████████████████████████████████████████████████████████| 37/37 [00:07<00:00,  4.88it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "36\n",
      "Empty DataFrame\n",
      "Columns: [count]\n",
      "Index: []\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "## test\n",
    "for data in tqdm(sorted(data_list, key=lambda x: int(x.split(\"_\")[-1].split(\".\")[0]))):\n",
    "    df_number = data.split(\"_\")[-1].split(\".\")[0]\n",
    "    df = pd.read_csv(data).interpolate()\n",
    "\n",
    "    new_cols = []\n",
    "    for col in df.columns:\n",
    "        df[col] = df[col].replace({' ': np.nan})\n",
    "        new_cols.append(re.sub('_$', '', re.sub('\\(.*\\)', '', col.replace(' ', '_'))))\n",
    "\n",
    "    # 칼럼명 수정\n",
    "    df.columns = new_cols\n",
    "#     print(df.columns)\n",
    "\n",
    "    # 사용할 열 선택 및 index 설정\n",
    "    df.drop(tr_del_list, axis=1, inplace=True)\n",
    "    df.set_index('datadate', drop=True, inplace=True)\n",
    "    \n",
    "    # 날씨 정보 평균으로 계산하기\n",
    "    df = avg_weather_cols(df)\n",
    "\n",
    "    # 사용할 열 선택 및 index 설정\n",
    "    df['해당일자_전체평균가격'].fillna(df['해당일자_전체평균가격'].mean(), inplace=True)\n",
    "    df['해당일자_전체거래물량'].fillna(df['해당일자_전체거래물량'].mean(), inplace=True)\n",
    "    df['하위가격_평균가'].fillna(df['하위가격_평균가'].mean(), inplace=True)\n",
    "    df['상위가격_평균가'].fillna(df['상위가격_평균가'].mean(), inplace=True)\n",
    "    df['하위가격_거래물량'].fillna(df['하위가격_거래물량'].mean(), inplace=True)\n",
    "    df['상위가격_거래물량'].fillna(df['상위가격_거래물량'].mean(), inplace=True)\n",
    "    df['일자별_도매가격_최대'].fillna(df['일자별_도매가격_최대'].mean(), inplace=True)\n",
    "    df['일자별_도매가격_평균'].fillna(df['일자별_도매가격_평균'].mean(), inplace=True)\n",
    "    df['일자별_도매가격_최소'].fillna(df['일자별_도매가격_최소'].mean(), inplace=True)\n",
    "    df['일자별_소매가격_최대'].fillna(df['일자별_소매가격_최대'].mean(), inplace=True)\n",
    "    df['일자별_소매가격_평균'].fillna(df['일자별_소매가격_평균'].mean(), inplace=True)\n",
    "    df['일자별_소매가격_최소'].fillna(df['일자별_소매가격_최소'].mean(), inplace=True)\n",
    "    \n",
    "#     df['주산지_평균_최대온도'].fillna(df['주산지_평균_최대온도'].mean(), inplace=True)\n",
    "#     df['주산지_평균_최저온도'].fillna(df['주산지_평균_최저온도'].mean(), inplace=True)\n",
    "#     df['주산지_평균_평균온도'].fillna(df['주산지_평균_평균온도'].mean(), inplace=True)\n",
    "#     df['주산지_평균_초기온도'].fillna(df['주산지_평균_초기온도'].mean(), inplace=True)\n",
    "#     df['주산지_평균_강수량'].fillna(df['주산지_평균_강수량'].mean(), inplace=True)\n",
    "\n",
    "    print(df_number)\n",
    "    all_null = pd.DataFrame(df.isnull().sum())\n",
    "    all_null.columns = ['count']\n",
    "#     print(all_null.iloc[0,0])\n",
    "    print(all_null.loc[all_null['count'] > 0])\n",
    "    print()\n",
    "    \n",
    "    df.to_csv(f'./results/{output_dir}/preprocessed_train_{df_number}.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "7207b1f8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['datadate', '단가', '거래량', '거래대금', '경매건수', '도매시장코드', '도매법인코드', '산지코드',\n",
       "       '해당일자_전체평균가격', '해당일자_전체거래물량', '하위가격_평균가', '상위가격_평균가', '하위가격_거래물량',\n",
       "       '상위가격_거래물량', '일자별_도매가격_최대', '일자별_도매가격_평균', '일자별_도매가격_최소', '일자별_소매가격_최대',\n",
       "       '일자별_소매가격_평균', '일자별_소매가격_최소', '수출중량', '수출금액', '수입중량', '수입금액', '무역수지',\n",
       "       '주산지_0_초기온도', '주산지_0_최대온도', '주산지_0_최저온도', '주산지_0_평균온도', '주산지_0_강수량',\n",
       "       '주산지_0_습도', '주산지_1_초기온도', '주산지_1_최대온도', '주산지_1_최저온도', '주산지_1_평균온도',\n",
       "       '주산지_1_강수량', '주산지_1_습도', '주산지_2_초기온도', '주산지_2_최대온도', '주산지_2_최저온도',\n",
       "       '주산지_2_평균온도', '주산지_2_강수량', '주산지_2_습도', '일자구분_중순', '일자구분_초순', '일자구분_하순',\n",
       "       '월구분_10월', '월구분_11월', '월구분_12월', '월구분_1월', '월구분_2월', '월구분_3월', '월구분_4월',\n",
       "       '월구분_5월', '월구분_6월', '월구분_7월', '월구분_8월', '월구분_9월'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mytest = pd.read_csv('./aT_data/data/train/train_0.csv')\n",
    "mytest.columns = new_cols\n",
    "mytest.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "279514ed",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  0%|                                                                                           | 0/37 [00:00<?, ?it/s]\n"
     ]
    },
    {
     "ename": "AttributeError",
     "evalue": "'str' object has no attribute 'plot'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "Input \u001b[1;32mIn [18]\u001b[0m, in \u001b[0;36m<cell line: 2>\u001b[1;34m()\u001b[0m\n\u001b[0;32m     16\u001b[0m     df\u001b[38;5;241m.\u001b[39mdrop(tr_del_list, axis\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m1\u001b[39m, inplace\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m)\n\u001b[0;32m     17\u001b[0m     df\u001b[38;5;241m.\u001b[39mset_index(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mdatadate\u001b[39m\u001b[38;5;124m'\u001b[39m, drop\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m, inplace\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m)\n\u001b[1;32m---> 19\u001b[0m     \u001b[43mdata\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mplot\u001b[49m(x\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mdatadate\u001b[39m\u001b[38;5;124m'\u001b[39m, y\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m해당일자_전체평균가격\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[0;32m     21\u001b[0m plt\u001b[38;5;241m.\u001b[39mshow()\n",
      "\u001b[1;31mAttributeError\u001b[0m: 'str' object has no attribute 'plot'"
     ]
    }
   ],
   "source": [
    "## test2\n",
    "for data in tqdm(sorted(data_list, key=lambda x: int(x.split(\"_\")[-1].split(\".\")[0]))):\n",
    "    df_number = data.split(\"_\")[-1].split(\".\")[0]\n",
    "    df = pd.read_csv(data).interpolate()\n",
    "\n",
    "    new_cols = []\n",
    "    for col in df.columns:\n",
    "        df[col] = df[col].replace({' ': np.nan})\n",
    "        new_cols.append(re.sub('_$', '', re.sub('\\(.*\\)', '', col.replace(' ', '_'))))\n",
    "\n",
    "    # 칼럼명 수정\n",
    "    df.columns = new_cols\n",
    "#     print(df.columns)\n",
    "\n",
    "    # 사용할 열 선택 및 index 설정\n",
    "    df.drop(tr_del_list, axis=1, inplace=True)\n",
    "    df.set_index('datadate', drop=True, inplace=True)\n",
    "    \n",
    "    data.plot(x='datadate', y='해당일자_전체평균가격')\n",
    "\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
