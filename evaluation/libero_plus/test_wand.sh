#!/bin/bash

echo "正在准备测试环境..."

# 1. 检查是否已经激活了 Conda 环境
if [ -z "$CONDA_PREFIX" ]; then
    echo "❌ 错误: 未检测到 CONDA 环境！请先执行: conda activate libero_plus_server"
    exit 1
fi

# 2. 最关键的一步：桥接 Conda 里的 ImageMagick 路径
export MAGICK_HOME="$CONDA_PREFIX"

# 3. 运行一小段内置的 Python 脚本来测试 Wand
echo "正在执行 'from wand.image import Image' ..."

python -c "
try:
    from wand.image import Image
    print('\n✅ [成功] 🎉 恭喜！Wand 成功找到了 Conda 环境里的 ImageMagick 引擎，完全没有报错！')
except Exception as e:
    print('\n❌ [失败] 导入 Wand 时发生错误，报错信息如下:')
    print('-' * 40)
    import traceback
    traceback.print_exc()
    print('-' * 40)
"