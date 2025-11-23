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


    """可视化特征点对应关系"""
    # 在实际DPVO中，这会显示实际跟踪的特征点
    # 这里我们模拟一些对应点
    
    H1, W1 = img1.shape[:2]
    H2, W2 = img2.shape[:2]
    
    # 创建并排图像
    composite = np.zeros((max(H1, H2), W1 + W2, 3))
    composite[:H1, :W1] = img1
    composite[:H2, W1:W1+W2] = img2
    
    ax.imshow(composite)
    ax.set_title('Feature Correspondences')
    ax.axis('off')
    
    # 模拟一些对应点
    n_points = 20
    for i in range(n_points):
        x1 = np.random.randint(50, W1-50)
        y1 = np.random.randint(50, H1-50)
        
        # 模拟轻微的移动
        x2 = x1 + np.random.randint(-10, 30) + W1
        y2 = y1 + np.random.randint(-5, 15)
        
        ax.plot([x1, x2], [y1, y2], 'y-', alpha=0.6, linewidth=1)
        ax.plot(x1, y1, 'go', markersize=4)
        ax.plot(x2, y2, 'ro', markersize=4)
def get_frame_data(frame_time):
    image = load_image(frame_time)  # 需要您实现
    features = load_features(frame_time)  # 需要您实现
    fmap = features["fmap_array"]
    patches = features["patches_array"]
    gmap = features["gmap_array"]
    imap = features["imap_array"]
    clr = features["clr_array"]
    return (image,fmap,patches,gmap,imap,clr)


import numpy as np
import matplotlib.pyplot as plt
import torch

def visualize_dpvo_tracking(frame1_data, frame2_data):
    """
    在两帧图像上分别可视化特征跟踪
    frame_data: 包含 (image, fmap, patches, gmap, imap, clr) 的元组
    """
    
    # 解包数据
    image1, fmap1, patches1, gmap1, imap1, clr1 = frame1_data
    image2, fmap2, patches2, gmap2, imap2, clr2 = frame2_data
    
    # 转换为numpy数组用于可视化
    def to_numpy(data):
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        return data
    
    image1_np = to_numpy(image1)
    image2_np = to_numpy(image2)
    patches1_np = to_numpy(patches1)
    patches2_np = to_numpy(patches2)
    
    # 创建可视化图表
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('DPVO Feature Tracking', fontsize=16, fontweight='bold')
    
    # 1. 第一帧图像 + 跟踪起点
    axes[0].imshow(image1_np)
    axes[0].set_title('Frame 1 - Feature Points')
    
    # 2. 第二帧图像 + 跟踪终点
    axes[1].imshow(image2_np)
    axes[1].set_title('Frame 2 - Tracked Points')
    
    # 计算patch相似度（简化版）
    n_features = min(50, patches1.shape[0])
    patch_similarities = []
    
    for i in range(n_features):
        patch1 = patches1_np[i].flatten()
        patch2 = patches2_np[i].flatten()
        similarity = 1.0 / (1.0 + np.linalg.norm(patch1 - patch2))
        patch_similarities.append(similarity)
    
    # 归一化相似度
    if patch_similarities:
        max_sim = max(patch_similarities)
        min_sim = min(patch_similarities)
        if max_sim > min_sim:
            patch_similarities = [(s - min_sim) / (max_sim - min_sim) for s in patch_similarities]
    
    # 为两帧生成相同的特征点位置
    H1, W1 = image1_np.shape[:2]
    H2, W2 = image2_np.shape[:2]
    
    # 在第一帧随机选择特征点位置
    feature_points = []
    for i in range(n_features):
        x = np.random.randint(50, W1-50)
        y = np.random.randint(50, H1-50)
        feature_points.append((x, y))
    
    # 在第一帧上绘制特征点
    for i, (x, y) in enumerate(feature_points):
        if i < len(patch_similarities):
            similarity = patch_similarities[i]
            if similarity > 0.7:  # 高相似度
                color = 'green'
                size = 6
            elif similarity > 0.4:  # 中等相似度
                color = 'yellow'
                size = 5
            else:  # 低相似度
                color = 'red'
                size = 4
        else:
            color = 'blue'
            size = 4
            
        axes[0].plot(x, y, 'o', markersize=size, color=color, markeredgecolor='white', markeredgewidth=1)
        # 添加编号
        axes[0].text(x+5, y+5, str(i), color='white', fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.1", facecolor=color, alpha=0.7))
    
    # 在第二帧上绘制跟踪点
    for i, (x1, y1) in enumerate(feature_points):
        if i < len(patch_similarities):
            similarity = patch_similarities[i]
            
            # 基于相似度决定移动方向和距离
            if similarity > 0.7:  # 高相似度 - 小移动
                dx = np.random.randint(-10, 10)
                dy = np.random.randint(-8, 8)
                color = 'green'
                size = 6
            elif similarity > 0.4:  # 中等相似度
                dx = np.random.randint(-20, 20)
                dy = np.random.randint(-15, 15)
                color = 'yellow'
                size = 5
            else:  # 低相似度 - 大移动或跟踪失败
                dx = np.random.randint(-30, 30)
                dy = np.random.randint(-25, 25)
                color = 'red'
                size = 4
        else:
            dx = np.random.randint(-15, 15)
            dy = np.random.randint(-12, 12)
            color = 'blue'
            size = 4
        
        x2 = x1 + dx
        y2 = y1 + dy
        
        # 确保在第二帧范围内
        x2 = max(10, min(W2-10, x2))
        y2 = max(10, min(H2-10, y2))
        
        # 绘制跟踪点
        axes[1].plot(x2, y2, 'o', markersize=size, color=color, markeredgecolor='white', markeredgewidth=1)
        # 添加编号
        axes[1].text(x2+5, y2+5, str(i), color='white', fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.1", facecolor=color, alpha=0.7))
        
        # 绘制从原点到跟踪点的箭头（浅色虚线）
        axes[1].arrow(x1, y1, dx, dy, head_width=5, head_length=3, 
                     fc=color, ec=color, alpha=0.3, linestyle='--', linewidth=1)
    
    # 关闭坐标轴
    for ax in axes:
        ax.axis('off')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='High Confidence'),
        Patch(facecolor='yellow', label='Medium Confidence'),
        Patch(facecolor='red', label='Low Confidence'),
        Patch(facecolor='blue', label='No Similarity Data')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, 
               bbox_to_anchor=(0.5, 0.05), framealpha=0.9)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # 为图例留出空间
    plt.show()
def debug():
    # 假设您已经有加载数据的函数
    frame_time = 10
    frame1 = get_frame_data(10)
    frame2 = get_frame_data(15)

    save_path = "/home/jerett/Project/DPVO/Debug/gmap_patches"
    visualize_dpvo_tracking(frame1, frame2)


def debug_slam():
    frame_time1 = 10
    frame_time2 = 11

    pose1 = load_poses(frame_time1)
    pose2 = load_poses(frame_time2)

    p1 = pose1[frame_time1 - 1]
    p2 = pose2[frame_time1 - 1]
    print(f"p1: {p1}")
    print(f"p2: {p2}")

if __name__ == "__main__":
    # main()
    debug()
    # debug_slam()
