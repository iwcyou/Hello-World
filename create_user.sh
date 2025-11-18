#!/bin/bash

# 检查是否为 root 权限
if [ "$(id -u)" -ne 0 ]; then
    echo "请使用 sudo 运行此脚本！"
    exit 1
fi

# 输入用户名
read -p "请输入要创建的新用户名: " USERNAME

# 输入密码（不会回显）
read -s -p "请输入用户密码: " PASSWORD
echo

# 创建用户
echo "正在创建用户：$USERNAME ..."
useradd -m "$USERNAME"

if [ $? -ne 0 ]; then
    echo "❌ 用户创建失败，可能用户已存在"
    exit 1
fi

# 设置密码
echo "$USERNAME:$PASSWORD" | chpasswd
echo "密码设置完成。"

# 创建 /data/username 和 /data1/username
for DIR in /data /data1; do
    TARGET="${DIR}/${USERNAME}"
    echo "正在创建目录：$TARGET ..."
    mkdir -p "$TARGET"
    chown "$USERNAME:$USERNAME" "$TARGET"
done

echo "目录创建并授权成功。"

echo "✔️ 用户 $USERNAME 创建完毕！"
