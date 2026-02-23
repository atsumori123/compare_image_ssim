import os
import cv2
import argparse
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ==========================
# 定義
# ==========================
SRC_HEADER_HEIGHT = 0		# 無視する比較元画像のヘッダー領域(px)
SRC_FOOTER_HEIGHT = 0		# 無視する比較元画像のフッター領域(px)
DST_HEADER_HEIGHT = 0		# 無視する比較先画像のヘッダー領域(px)
DST_FOOTER_HEIGHT = 0		# 無視する比較先画像のフッター領域(px)
MARGIN_THRESHOLD = 250		# 余白のトリミング閾値
DIFF_THRESHOLD = 200		# 差分の閾値。150～230が推奨。(値が小さいほど小さな差分を検出しやすくなる)
HEATMAP_OVERLAY = 0			# ヒートマップのオーバレイ
HEATMAP_ALPHA = 0.4			# ヒートマップの透明度

# ==========================
# 余白を自動トリミング
# ==========================
def trim_margin(img, threshold=250):
	# 色ではなく明暗で判定するため白黒画像(2値化)に変換する
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_, th = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

	# 余白以外(文字や図)を抽出。(255 - th)で、
	# 文字・図→白(255)
	# 余白→黒(0)
	coords = cv2.findNonZero(255 - th)
	if coords is None:
		return img

	# 文字、図が存在する最小の矩形を計算し、その領域を切り出す
	x, y, w, h = cv2.boundingRect(coords)
	return img[y:y+h, x:x+w], (x, y, w, h)


# ==========================
# ヘッダー・フッター除去
# ==========================
def crop_header_footer(img, header_height, footer_height):
	# 画像の高さを取得
	h = img.shape[0]

	# ヘッダー領域以降からフッターまでを切り出す
	return img[header_height:h-footer_height, :], header_height


# ==========================
# SSIM 比較（フォント差異吸収のためぼかし）
# ==========================
def compare_images(img1, img2):
	# グレースケール化
	g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	# フォント差異吸収のため軽くぼかす
	g1 = cv2.GaussianBlur(g1, (5, 5), 0)
	g2 = cv2.GaussianBlur(g2, (5, 5), 0)

	# SSIM
	score, diff = ssim(g1, g2, full=True)
	diff = (diff * 255).astype("uint8")

	return score, diff


# ==========================
# 差分領域を赤枠で囲む（元画像に描画）
# ==========================
def highlight_diff(outimg, diff, trim_info, header_offset, threshold=DIFF_THRESHOLD):
	(trim_x, trim_y, _, _) = trim_info

	# SSIMの差分マップ(0～255)を2値化して差分領域を抽出
	# 差分がある部分を白くする
	# ノイズ除去のため膨張処理(dilate)
	_, th = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY_INV)
	th = cv2.dilate(th, None, iterations=2)

	# 差分領域の輪郭を取得
	contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# 差分面積を計算
	# thは差分がある部分が白(255)の二値画像
	area = int(np.count_nonzero(th))

	# オリジナル画像のコピー
	output = outimg.copy()
	has_diff = False

	for cnt in contours:
		x, y, w, h = cv2.boundingRect(cnt)

		# 前処理画像 → 元画像の座標に変換
		orig_x = x + trim_x
		orig_y = y + trim_y + header_offset

		# 元画像に赤枠を描画する
		cv2.rectangle(output, (orig_x, orig_y), (orig_x + w, orig_y + h), (0, 0, 255), 3)
		has_diff = True

	return output, has_diff, area

# ==========================
# ヒートマップの作成
# ==========================
def generate_heatmap(outimg, diff, trim_info, header_offset, alpha=HEATMAP_ALPHA):
	# outimg		: 元のフル画像
	# diff			: SSIM の差分マップ（0〜255）
	# trim_info		: (trim_x, trim_y, w, h) 余白トリミングのオフセット
	# header_offset	: ヘッダー除去のオフセット
	# alpha			: ヒートマップの透明度

	(trim_x, trim_y, _, _) = trim_info

	# SSIM diff は「似ているほど白」なので反転して差分強度にする
	diff_inv = 255 - diff

	# カラーマップ化（COLORMAP_JET が最も視覚的にわかりやすい）
	heatmap = cv2.applyColorMap(diff_inv, cv2.COLORMAP_JET)

	# 前処理画像 → 元画像の座標に合わせるためのキャンバスを作成
	h, w = outimg.shape[:2]
	heatmap_canvas = np.zeros_like(outimg)

	# ヒートマップを元画像の該当位置に貼り付け
	y1 = trim_y + header_offset
	y2 = y1 + heatmap.shape[0]
	x1 = trim_x
	x2 = x1 + heatmap.shape[1]

	heatmap_canvas[y1:y2, x1:x2] = heatmap

	# 元画像とヒートマップを合成（透明度 alpha）
	overlay = cv2.addWeighted(outimg, 1 - alpha, heatmap_canvas, alpha, 0)

	return overlay


# ==========================
# 差分強度の数値化
# ==========================
def compute_diff_metrics(diff, threshold):
	diff_inv = 255 - diff

	diff_mask = diff_inv > threshold
	diff_ratio = np.count_nonzero(diff_mask) / diff_inv.size

	mean_diff = float(np.mean(diff_inv))
	max_diff = int(np.max(diff_inv))

	return diff_ratio, mean_diff, max_diff


# ==========================
# メイン処理
# ==========================
def main(src_dir, dst_dir, out_dir="output"):
	# 比較対象ディレクトリ配下にあるpngファイルをソートして取得
	src_files = sorted([f for f in os.listdir(src_dir) if f.endswith(".png")])

	# pngファイル数分比較を繰り返す
	for filename in src_files:
		# 比較元、比較先ファイルパスの取得
		src_path = os.path.join(src_dir, filename)
		dst_path = os.path.join(dst_dir, filename)

		# ファイル有無をチェック。比較先ディレクトリに同じ名前のファイルが無かったらスキップ
		if not os.path.exists(dst_path):
			print(f"[SKIP] {filename} が比較先にありません")
			continue

		# 画像読み込み
		# 比較先画像は差分描画用の分も設定しておく
		img1 = cv2.imread(src_path)
		img2 = outimg = cv2.imread(dst_path)

		# 余白除去
		img1, trim1 = trim_margin(img1)
		img2, trim2 = trim_margin(img2)

		# ヘッダー・フッター除去
		img1, header_offset1 = crop_header_footer(img1, SRC_HEADER_HEIGHT, SRC_FOOTER_HEIGHT)
		img2, header_offset2 = crop_header_footer(img2, DST_HEADER_HEIGHT, SRC_FOOTER_HEIGHT)

		# サイズを小さい画像に合わせて揃える
		h = min(img1.shape[0], img2.shape[0])
		w = min(img1.shape[1], img2.shape[1])
		img1 = img1[:h, :w]
		img2 = img2[:h, :w]

		# SSIM 比較
		# scoreの解釈イメージ
		#	------------------------------------------------------------------------
		#	 SSIM			差分率		解釈
		#	------------------------------------------------------------------------
		#	0.98～1.00		0～1%		ほぼ同じ。フォント差異や微細なノイズ程度
		#	0.90〜0.98		1～5%		少し違う。文章の一部変更、図の一部変更
		#	0.70〜0.90		5〜20%		明確に違う。図の大幅変更、段落追加
		#	0.70 未満		20%以上		かなり違う。別のページレベルの差異
		#
		# ※SSIMの値はあくまで参考値。SSIMで差分あり/なしを判断するのは危険。以下理由。
		#   ・局所的な差異を見逃す
		#   ・フォント差異でSSIMが下がる
		#   ・ページ全体の構造に依存する
		score, diff = compare_images(img1, img2)

		# 差分の強度を数値化
		diff_ratio, mean_diff, max_diff = compute_diff_metrics(diff, DIFF_THRESHOLD)

		if HEATMAP_OVERLAY:
			# ヒートマップをオーバレイ
			# 青：ほぼ差分なし
			# 緑：わずかな差分
			# 黄色：中程度の差分
			# 赤：大きな差分（文章変更・図の変更など）
			outimg = generate_heatmap(outimg, diff, trim2, header_offset2)

		# 差分を元画像に描画
		highlighted, has_diff, area = highlight_diff(outimg, diff, trim2, header_offset2)

		if has_diff:
			out_path = os.path.join(out_dir, f"diff_{filename}")
			cv2.imwrite(out_path, highlighted)
			print(f"[DIFF] {filename}  SSIM={score:.4f}  diff={diff_ratio*100:.2f}%  差分面積={area}px  → {out_path}")
		else:
			print(f"[ OK ] {filename}  SSIM={score:.4f}  diff={diff_ratio*100:.2f}%  差分面積={area}px")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("source", help="source diretory")
	parser.add_argument("destination", help="destinatin diretory")
	parser.add_argument("-sh", help="source image header height (px) (default:0)")
	parser.add_argument("-sf", help="source image footer height (px) (default:0)")
	parser.add_argument("-dh", help="destinatin image header height (px) (default:0)")
	parser.add_argument("-df", help="destinatin image footer height (px) (default:0)")
	parser.add_argument("-threshold", help="diff threshold (0-255) (default:200)")
	parser.add_argument("-heatmap", help="overlay heatmap (0,1) (default:0)")
	args = parser.parse_args()

	# 比較結果出力ディレクトリの作成
	os.makedirs("output", exist_ok=True)

	# 比較元画像のヘッダとフッターの高さ
	if args.sh: SRC_HEADER_HEIGHT = int(args.sh)
	if args.sf: SRC_FOOTER_HEIGHT = int(args.sf)

	# 比較先画像のヘッダとフッターの高さ
	if args.sh: DST_HEADER_HEIGHT = int(args.dh)
	if args.sf: DST_FOOTER_HEIGHT = int(args.df)

	# 差分の閾値
	if args.threshold: DIFF_THRESHOLD = int(args.threshold)

	# ヒートマップ
	if args.heatmap: HEATMAP_OVERLAY = 1

	main(args.source, args.destination, "output")


