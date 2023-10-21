### 1. 将 conda 的环境添加到 notebook 中

```shell
conda activate 虚拟环境名
conda install ipykernel
python -m ipykernel install --user --name 虚拟环境名 --display-name "自定义名字"
jupyter kernelspec list   #查看当前notebook中所具有的kernel
```

在该虚拟环境中还需要重新安装jupyter notebook

```bash
pip install jupyter notebook
```

### 2. 代码自动填充 Auto-fill

```bash
pip install jupyter_contrib_nbextensions

jupyter contrib nbextension install --user

pip install --user jupyter_nbextensions_configurator 

jupyter nbextensions_configurator enable --user

jupyter nbextension enable
```