from dpvo.debug_utils import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.manifold import TSNE

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def fmap_vis(fmap):
    # 1. 数据重塑：将三维张量展开为二维矩阵 (像素数, 特征数)
    pixels = fmap.reshape(
        -1, fmap.shape[-1]
    )  # 形状变为 (132 * 240, 128) -> (31680, 128)

    # 2. 数据标准化（可选，但强烈推荐）：去除均值，缩放至单位方差
    scaler = StandardScaler()
    pixels_scaled = scaler.fit_transform(pixels)

    # 3. 执行PCA降维，提取前3个主成分（对应RGB三通道）
    pca = PCA(n_components=3)
    pixels_pca = pca.fit_transform(pixels_scaled)  # 形状变为 (31680, 3)

    # 4. 将降维后的数据重塑回图像格式 (132, 240, 3)
    rgb_image = pixels_pca.reshape(
        fmap.shape[0], fmap.shape[1], 3
    )  # 形状变为 (132, 240, 3)

    # 5. 将主成分值归一化到 [0, 255] 的整数范围，以便显示为RGB图像
    # 由于PCA后的值有正有负，我们使用最小-最大归一化
    def normalize_to_uint8(data):
        """将数据线性映射到0-255并转换为uint8类型"""
        data_min = data.min(axis=(0, 1), keepdims=True)  # 计算每个通道的最小值
        data_max = data.max(axis=(0, 1), keepdims=True)  # 计算每个通道的最大值
        # 避免除以零，如果最大值等于最小值，则该通道为常数
        normalized = np.where(
            data_max != data_min, 255 * (data - data_min) / (data_max - data_min), 0
        )
        return normalized.astype(np.uint8)

    rgb_image_normalized = normalize_to_uint8(rgb_image)
    upscaled_image = cv2.resize(
        rgb_image_normalized,
        None,  # 不直接指定尺寸，通过fx和fy参数控制缩放
        fx=4,  # 水平方向放大4倍
        fy=4,  # 垂直方向放大4倍
        interpolation=cv2.INTER_CUBIC,  # 使用双三次插值，效果较好
    )
    cv2.imshow("fmap", upscaled_image)


def gmap_vis(gmap):
    # 1. 数据重塑：将每个补丁的 (96, 3, 3) 特征展平
    # 转换维度，将补丁数(128)放在最前面
    gmap_transposed = gmap.transpose(1, 0, 2, 3)  # 新形状: (128, 96, 3, 3)
    gmap_flat = gmap_transposed.reshape(
        gmap_transposed.shape[0], -1
    )  # 形状: (128, 96 * 3 * 3)

    # 使用t-SNE进行非线性降维，有时能揭示PCA难以发现的局部结构[9,10](@ref)
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    gmap_tsne_3d = tsne.fit_transform(gmap_flat)

    # 创建交互式3D散点图
    fig = px.scatter_3d(
        x=gmap_tsne_3d[:, 0], y=gmap_tsne_3d[:, 1], z=gmap_tsne_3d[:, 2], opacity=0.7
    )
    fig.update_layout(title="gmap 的 t-SNE 3D 可视化")
    fig.show()


def visualize_single_frame(image, features):
    """
    可视化单帧图像及其所有相关特征张量

    参数:
        image: 原始图像数组 (H, W, 3)
        features: 包含fmap, gmap, imap, patches, clr等特征的字典
    """

    # 1. 首先可视化原始图像
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "SimHei",
    ]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    plt.figure(figsize=(15, 10))

    # 原始图像
    plt.subplot(2, 3, 1)
    # 注意：如果图像是BGR格式，需要转换为RGB显示
    if image is not None:
        display_image = (
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if len(image.shape) == 3 and image.shape[2] == 3
            else image
        )
        plt.imshow(display_image)
        plt.title("原始图像")
        plt.axis("off")
    else:
        plt.text(0.5, 0.5, "无图像数据", ha="center", va="center")
        plt.title("原始图像 (无数据)")
        plt.axis("off")

    # 2. 可视化fmap (特征图)
    plt.subplot(2, 3, 2)
    if "fmap_array" in features:
        fmap = features["fmap_array"]
        print(f"fmap 原始形状: {fmap.shape}")

        # 将通道维度移到最后 (C, H, W) -> (H, W, C)
        fmap_transposed = np.transpose(fmap, (1, 2, 0))
        print(f"fmap 转置后形状: {fmap_transposed.shape}")

        # 使用PCA将高维特征降维到3通道用于可视化
        h, w, c = fmap_transposed.shape
        fmap_flat = fmap_transposed.reshape(-1, c)

        # 执行PCA降维
        pca = PCA(n_components=3)
        fmap_pca = pca.fit_transform(fmap_flat)

        # 归一化到[0, 1]范围
        fmap_pca_normalized = (fmap_pca - fmap_pca.min()) / (
            fmap_pca.max() - fmap_pca.min()
        )
        fmap_vis = fmap_pca_normalized.reshape(h, w, 3)

        plt.imshow(fmap_vis)
        plt.title(
            f"fmap特征 (PCA降维)\n解释方差: {pca.explained_variance_ratio_.sum():.2f}"
        )
        plt.axis("off")
    else:
        plt.text(0.5, 0.5, "无fmap数据", ha="center", va="center")
        plt.title("fmap特征")
        plt.axis("off")

    # 3. 可视化patches (图像块)
    plt.subplot(2, 3, 3)
    if "patches_array" in features:
        patches = features["patches_array"]
        print(f"patches形状: {patches.shape}")

        # 创建patches的网格可视化 (假设patches形状为 [n_patches, height, width, channels])
        n_patches = patches.shape[0]
        grid_size = int(np.ceil(np.sqrt(n_patches)))

        # 创建一个大的画布来显示所有patches
        patch_h, patch_w = patches.shape[1], patches.shape[2]
        canvas = np.zeros((grid_size * patch_h, grid_size * patch_w, 3))

        for i in range(min(n_patches, grid_size * grid_size)):
            row = i // grid_size
            col = i % grid_size
            patch = patches[i]

            # 归一化patches到[0, 1]范围
            if patch.max() > 1:
                patch = patch / 255.0

            canvas[
                row * patch_h : (row + 1) * patch_h, col * patch_w : (col + 1) * patch_w
            ] = patch

        plt.imshow(canvas)
        plt.title(f"Patches可视化 (共{n_patches}个)")
        plt.axis("off")
    else:
        plt.text(0.5, 0.5, "无patches数据", ha="center", va="center")
        plt.title("Patches")
        plt.axis("off")

    # 4. 可视化clr (颜色信息)
    plt.subplot(2, 3, 4)
    if "clr_array" in features:
        clr = features["clr_array"]
        print(f"clr形状: {clr.shape}")

        # 显示颜色条
        n_colors = clr.shape[0]
        color_swatch = np.ones((50, n_colors * 10, 3), dtype=np.uint8)

        for i in range(n_colors):
            color_swatch[:, i * 10 : (i + 1) * 10] = clr[i]

        plt.imshow(color_swatch)
        plt.title(f"颜色信息 (共{n_colors}种)")
        plt.axis("off")
    else:
        plt.text(0.5, 0.5, "无clr数据", ha="center", va="center")
        plt.title("颜色信息")
        plt.axis("off")

    # 5. 可视化gmap (上下文特征图) - 显示统计信息
    plt.subplot(2, 3, 5)
    if "gmap_array" in features:
        gmap = features["gmap_array"]
        print(f"gmap形状: {gmap.shape}")

        # 显示gmap的统计信息
        plt.text(0.1, 0.7, f"gmap形状: {gmap.shape}", fontsize=12)
        plt.text(
            0.1, 0.5, f"数值范围: [{gmap.min():.3f}, {gmap.max():.3f}]", fontsize=12
        )
        plt.text(
            0.1, 0.3, f"均值: {gmap.mean():.3f}, 标准差: {gmap.std():.3f}", fontsize=12
        )
        plt.title("gmap特征统计")
        plt.axis("off")
    else:
        plt.text(0.5, 0.5, "无gmap数据", ha="center", va="center")
        plt.title("gmap特征")
        plt.axis("off")

    # 6. 可视化imap (索引映射/隐藏状态) - 显示统计信息
    plt.subplot(2, 3, 6)
    if "imap_array" in features:
        imap = features["imap_array"]
        print(f"imap形状: {imap.shape}")

        # 显示imap的统计信息
        plt.text(0.1, 0.7, f"imap形状: {imap.shape}", fontsize=12)
        plt.text(
            0.1, 0.5, f"数值范围: [{imap.min():.3f}, {imap.max():.3f}]", fontsize=12
        )
        plt.text(
            0.1, 0.3, f"均值: {imap.mean():.3f}, 标准差: {imap.std():.3f}", fontsize=12
        )
        plt.title("imap特征统计")
        plt.axis("off")
    else:
        plt.text(0.5, 0.5, "无imap数据", ha="center", va="center")
        plt.title("imap特征")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    frame_time = 50
    image = load_image(frame_time)
    # cv2.imshow("image",image)

    intrinsic = load_intrinsics(frame_time)
    features = load_features(frame_time)

    ##### fmap
    fmap = features["fmap_array"]
    fmap = np.transpose(fmap, (1, 2, 0))
    # fmap_vis(fmap)

    ##### gmap
    gmap = features["gmap_array"]
    # gmap_vis(gmap)

    ##### imap
    imap = features["imap_array"]
    print(f"fmap: {fmap.shape},gmap: {gmap.shape},imap: {imap.shape}")

    ##### patch
    patches = features["patches_array"]

    ##### clr
    clr = features["clr_array"]
    print(
        f"image: {image.shape},fmap: {fmap.shape},gmap: {gmap.shape},imap: {imap.shape},patches: {patches.shape},clr: {clr.shape}"
    )

    visualize_single_frame(image, features)
    # cv2.waitKey(-1)


import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


def visualize_dpvo_features(image, gmap, patches, save_path, frame_time=50):
    """
    将DPVO的gmap和patches特征可视化在原始图像上，并保存结果

    参数:
        image: 原始图像数组 (H, W, 3)
        gmap: gmap特征数组 (96, 128, 3, 3) - 假设维度为 (补丁数, 特征通道数, 空间高, 空间宽)
        patches: patches数组 (96, 3, 3, 3) - 假设维度为 (补丁数, 通道, 高, 宽)
        save_path: 结果保存路径
        frame_time: 时间帧编号，用于文件名
    """
    # 创建保存目录
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"开始处理帧 {frame_time}")
    print(f"输入维度: image={image.shape}, gmap={gmap.shape}, patches={patches.shape}")

    # 修复核心问题：确保gmap和patches的补丁数一致
    n_patches = patches.shape[0]  # 应为96
    if gmap.shape[0] != n_patches:
        print(
            f"警告: gmap补丁数({gmap.shape[0]})与patches补丁数({n_patches})不匹配，将使用最小公共补丁数"
        )
        n_patches = min(gmap.shape[0], n_patches)
        gmap = gmap[:n_patches]  # 截取前n_patches个补丁
        patches = patches[:n_patches]

    print(f"使用补丁数: {n_patches}")

    # 1. 计算每个补丁的gmap特征强度
    # gmap形状: (n_patches, 特征通道数, 3, 3)
    # 对每个补丁，聚合所有特征通道和空间维度，得到特征强度
    intensities = np.mean(gmap, axis=(1, 2, 3))  # 形状: (n_patches,)
    print(
        f"特征强度计算完成: {intensities.shape}, 范围[{intensities.min():.3f}, {intensities.max():.3f}]"
    )

    # 2. 生成补丁坐标（均匀分布在图像中央区域）
    h, w = image.shape[0], image.shape[1]
    grid_size = int(np.ceil(np.sqrt(n_patches)))

    # 在图像中央80%区域生成均匀网格
    margin_x, margin_y = w // 10, h // 10
    x_coords = np.linspace(margin_x, w - margin_x, grid_size).astype(int)
    y_coords = np.linspace(margin_y, h - margin_y, grid_size).astype(int)
    xx, yy = np.meshgrid(x_coords, y_coords)
    coordinates = list(zip(xx.flatten(), yy.flatten()))
    coordinates = coordinates[:n_patches]  # 确保数量匹配

    x = [coord[0] for coord in coordinates]
    y = [coord[1] for coord in coordinates]

    print(f"生成坐标完成: {len(x)}个点")

    # 3. 创建多种可视化
    create_combined_visualization(
        image, gmap, patches, intensities, x, y, save_path, frame_time
    )
    create_heatmap_overlay(image, intensities, x, y, save_path, frame_time)
    create_patch_detail_visualization(
        image, patches, intensities, x, y, save_path, frame_time
    )

    # 4. 保存原始数据供进一步分析
    data_path = save_path / f"dpvo_data_frame{frame_time:06d}.npz"
    np.savez(
        data_path,
        image=image,
        gmap=gmap,
        patches=patches,
        intensities=intensities,
        coordinates=np.array(coordinates),
    )
    print(f"原始数据已保存: {data_path}")

    print(f"帧 {frame_time} 的可视化完成，结果保存至: {save_path}")


def create_combined_visualization(
    image, gmap, patches, intensities, x, y, save_path, frame_time
):
    """创建综合可视化：原始图像 + 特征强度散点图 + 补丁示例"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 左侧：原始图像与特征强度散点
    display_img = (
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(image.shape) == 3 and image.shape[2] == 3
        else image
    )

    ax1.imshow(display_img)
    scatter1 = ax1.scatter(
        x,
        y,
        c=intensities,
        cmap="viridis",
        s=60,
        alpha=0.8,
        edgecolors="white",
        linewidth=1.5,
    )
    ax1.set_title(
        f"帧 {frame_time} - GMap特征强度分布\n(补丁数: {len(intensities)})", fontsize=14
    )
    ax1.axis("off")
    plt.colorbar(scatter1, ax=ax1, label="特征强度", fraction=0.046, pad=0.04)

    # 右侧：特征强度统计与补丁示例
    # 上子图：特征强度分布直方图
    ax2_upper = plt.subplot(2, 2, 2)
    ax2_upper.hist(intensities, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
    ax2_upper.set_xlabel("特征强度")
    ax2_upper.set_ylabel("频数")
    ax2_upper.set_title("特征强度分布")
    ax2_upper.grid(True, alpha=0.3)

    # 下子图：显示几个示例补丁
    ax2_lower = plt.subplot(2, 2, 4)
    n_examples = min(9, len(patches))
    example_indices = np.linspace(0, len(patches) - 1, n_examples, dtype=int)

    # 创建补丁网格
    patch_size = patches.shape[2]  # 假设为3
    grid_size = int(np.ceil(np.sqrt(n_examples)))
    canvas = np.zeros((grid_size * patch_size, grid_size * patch_size, 3))

    for i, idx in enumerate(example_indices):
        row = i // grid_size
        col = i % grid_size
        patch = patches[idx]

        # 归一化补丁到[0,1]范围
        if patch.max() > 1:
            patch = patch / 255.0
        elif patch.min() < 0:
            patch = (patch - patch.min()) / (patch.max() - patch.min())

        # 确保补丁是HWC格式
        if patch.shape[0] == 3:  # CHW格式
            patch = np.transpose(patch, (1, 2, 0))

        canvas[
            row * patch_size : (row + 1) * patch_size,
            col * patch_size : (col + 1) * patch_size,
        ] = patch

    ax2_lower.imshow(canvas)
    ax2_lower.set_title(f"示例补丁 (共{len(patches)}个)")
    ax2_lower.set_xticks([])
    ax2_lower.set_yticks([])

    # 添加强度文本信息
    stats_text = f"""特征强度统计:
均值: {intensities.mean():.4f}
标准差: {intensities.std():.4f}
最大值: {intensities.max():.4f}
最小值: {intensities.min():.4f}
补丁数: {len(intensities)}"""

    ax2_upper.text(
        0.02,
        0.98,
        stats_text,
        transform=ax2_upper.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontsize=10,
    )

    plt.tight_layout()

    # 保存综合可视化
    fig_path = save_path / f"combined_visualization_frame{frame_time:06d}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"综合可视化已保存: {fig_path}")


def create_heatmap_overlay(image, intensities, x, y, save_path, frame_time):
    """创建热力图叠加可视化"""
    h, w = image.shape[0], image.shape[1]

    # 创建热力图画布
    heatmap = np.zeros((h, w), dtype=np.float32)

    # 在每个补丁位置创建高斯分布
    for i, (center_x, center_y) in enumerate(zip(x, y)):
        intensity = intensities[i]

        # 创建高斯核
        kernel_size = 31
        y_range, x_range = np.ogrid[
            -kernel_size // 2 : kernel_size // 2 + 1,
            -kernel_size // 2 : kernel_size // 2 + 1,
        ]
        gaussian_patch = (
            np.exp(-(x_range**2 + y_range**2) / (2 * (kernel_size // 6) ** 2))
            * intensity
        )

        # 将高斯分布添加到热力图上
        y_start = max(0, center_y - kernel_size // 2)
        y_end = min(h, center_y + kernel_size // 2 + 1)
        x_start = max(0, center_x - kernel_size // 2)
        x_end = min(w, center_x + kernel_size // 2 + 1)

        patch_h = y_end - y_start
        patch_w = x_end - x_start

        if patch_h > 0 and patch_w > 0:
            gy_start = max(0, kernel_size // 2 - (center_y - y_start))
            gy_end = min(kernel_size, kernel_size // 2 + (y_end - center_y))
            gx_start = max(0, kernel_size // 2 - (center_x - x_start))
            gx_end = min(kernel_size, kernel_size // 2 + (x_end - center_x))

            gaussian_cropped = gaussian_patch[gy_start:gy_end, gx_start:gx_end]
            heatmap[y_start:y_end, x_start:x_end] += gaussian_cropped

    # 平滑热力图
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)

    # 归一化
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # 创建叠加可视化
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    display_img = (
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(image.shape) == 3 and image.shape[2] == 3
        else image
    )

    ax.imshow(display_img)
    im = ax.imshow(heatmap, cmap="jet", alpha=0.6, extent=[0, w, h, 0])
    ax.set_title(f"帧 {frame_time} - GMap特征热力图叠加", fontsize=14)
    ax.axis("off")
    plt.colorbar(im, ax=ax, label="特征强度", fraction=0.046, pad=0.04)

    # 保存热力图
    heatmap_path = save_path / f"heatmap_overlay_frame{frame_time:06d}.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"热力图叠加已保存: {heatmap_path}")


def create_patch_detail_visualization(
    image, patches, intensities, x, y, save_path, frame_time
):
    """创建补丁详细信息可视化"""
    n_patches = len(patches)
    n_to_show = min(16, n_patches)  # 最多显示16个补丁

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()

    display_img = (
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(image.shape) == 3 and image.shape[2] == 3
        else image
    )

    for i in range(n_to_show):
        ax = axes[i]

        # 显示补丁在图像中的位置
        ax.imshow(display_img)
        ax.scatter(
            x[i], y[i], color="red", s=100, marker="o", edgecolors="white", linewidth=2
        )
        ax.set_xlim(x[i] - 50, x[i] + 50)
        ax.set_ylim(y[i] + 50, y[i] - 50)  # 注意y轴方向
        ax.set_title(f"补丁 #{i}\n强度: {intensities[i]:.3f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # 隐藏多余的子图
    for j in range(n_to_show, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"帧 {frame_time} - 补丁位置详情 (前{n_to_show}个补丁)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存补丁详情图
    detail_path = save_path / f"patch_details_frame{frame_time:06d}.png"
    plt.savefig(detail_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"补丁详情图已保存: {detail_path}")


# 使用示例
def debug():
    # 假设您已经有加载数据的函数
    frame_time = 50
    image = load_image(frame_time)  # 需要您实现
    features = load_features(frame_time)  # 需要您实现
    patches = features["patches_array"]
    gmap = features["gmap_array"]
    imap = features["imap_array"]
    clr = features["clr_array"]

    print(
        f"image: {image.shape},patches: {patches.shape},gmap: {gmap.shape},imap: {imap.shape},clr: {clr.shape}"
    )
    save_path = "/home/jerett/Project/DPVO/Debug/gmap_patches"

    # 调用可视化函数
    # visualize_dpvo_features(image, gmap, patches, save_path, frame_time)


if __name__ == "__main__":
    # main()
    debug()
