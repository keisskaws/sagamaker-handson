# BYOL用のカスタムDockerfile
FROM python:3.8-slim

# 必要なシステムパッケージをインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pythonライブラリをインストール
COPY requirements.txt /opt/ml/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/requirements.txt

# カスタムライブラリをインストール（例：独自のMLライブラリ）
COPY custom_ml_lib/ /opt/ml/custom_ml_lib/
RUN cd /opt/ml/custom_ml_lib && pip install -e .

# トレーニングスクリプトをコピー
COPY train.py /opt/ml/code/train.py

# SageMakerの環境変数設定
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

# 作業ディレクトリ設定
WORKDIR /opt/ml/code

# エントリーポイント設定
ENTRYPOINT ["python", "train.py"]
