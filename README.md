### Fashion Mnist

#### 実行方法
```
python3 fashion_mnist.py "実行引数"
```
- 実行ファイル:&ensp;fashion_mnist.py
- 実行引数:&ensp;learn(学習) or verification(検証)
- Accuracy:&ensp;0.9024
<br></br>

### 学習済みパラメータ

- https://drive.google.com/drive/folders/1_DFCl6D75DYAm524k0C_i8JyVeZYKvdf?usp=sharing

&emsp;&emsp;&ensp;※pklファイルをfiles/params/ディレクトリに保存して下さい。
<br></br>

### 正答率の変動

- https://drive.google.com/drive/folders/1RcxULMkVJK7ZrlN4LdxW8Y7qPhJkjIVl?usp=sharing
<br></br>

### 利用方法

#### containerの起動

``` 
docker-compose up -d
```

#### pythonファイルの実行

```
docker-compose exec deeplearning "実行コマンド"
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
