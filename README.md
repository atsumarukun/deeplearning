### Fashion Mnist

- fashion_mnist.py
- Accuracy: 0.90
<br></br>

### 学習済みパラメータ

- https://drive.google.com/drive/folders/1_DFCl6D75DYAm524k0C_i8JyVeZYKvdf?usp=sharing

&emsp;&emsp;&ensp;※pklファイルをfiles/params/ディレクトリに保存して下さい。
<br></br>

### 利用方法

#### containerの起動

``` 
docker-compose up -d
```

#### pythonファイルの実行

```
docker-compose exec deeplearning python3 "file name"
```

#### containerに入る

```
docker-compose exec deeplearning bash
```

#### containerの停止及び削除

```
docker-compose down --rmi all
```

#### リファレンス

- https://docs.docker.jp/compose/toc.html
<br></br>

### 作業環境

- Arch Linux
- Docker (Ubuntu:22.04)
<br></br>