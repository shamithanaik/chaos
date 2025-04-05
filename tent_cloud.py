import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import time
from scipy.stats import entropy as scipy_entropy

# ------------------ Chaotic Maps ------------------
def bernoulli_map(size, seed):
    x = seed
    sequence = []
    for _ in range(size):
        x = (2 * x) % 1
        sequence.append(x)
    return np.array(sequence)

def tent_map(size, seed):
    x = seed
    sequence = []
    for _ in range(size):
        if x < 0.5:
            x = 2 * x
        else:
            x = 2 * (1 - x)
        sequence.append(x)
    return np.array(sequence)

# ------------------ Bitwise Permutation ------------------
def bitwise_permute(image):
    H, W, C = image.shape
    permuted_image = np.zeros_like(image)
    for ch in range(C):
        for bit in range(8):
            bitplane = ((image[:, :, ch] >> bit) & 1).astype(np.uint8)
            bitplane = np.roll(bitplane, shift=np.random.randint(1, H * W), axis=1)
            permuted_image[:, :, ch] |= (bitplane << bit)
    return permuted_image

# ------------------ Bitplane XOR with Tent Sequence ------------------
def bitplane_xor_with_tent(image, key):
    H, W, C = image.shape
    xor_image = np.copy(image)
    for ch in range(C):
        for bit in range(8):
            tent_seq = (tent_map(H * W, key + bit + ch) > 0.5).astype(np.uint8).reshape(H, W)
            bitplane = ((xor_image[:, :, ch] >> bit) & 1)
            bitplane ^= tent_seq
            mask = 0xFF ^ (1 << bit)
            xor_image[:, :, ch] &= mask
            xor_image[:, :, ch] |= (bitplane << bit)
    return xor_image

# ------------------ Preprocessing with Bernoulli XOR ------------------
def preprocess_with_bernoulli(image, key):
    H, W, C = image.shape
    chaotic_seq = (bernoulli_map(H * W, key) * 255).astype(np.uint8).reshape(H, W)
    processed = np.copy(image)
    for ch in range(C):
        processed[:, :, ch] ^= chaotic_seq
    return processed

# ------------------ Full Scrambling ------------------
def full_scramble(image, key):
    H, W, C = image.shape
    row_perm = np.argsort(bernoulli_map(H, key))
    col_perm = np.argsort(bernoulli_map(W, key))
    scrambled = np.copy(image)
    for ch in range(C):
        scrambled[:, :, ch] = scrambled[row_perm, :, ch]
        scrambled[:, :, ch] = scrambled[:, col_perm, ch]
    return scrambled, row_perm, col_perm

# ------------------ Blockwise Diffusion ------------------
def blockwise_diffuse(image, key):
    H, W, C = image.shape
    B = 8
    block_seq = (tent_map(B * B, key) * 255).astype(np.uint8)
    global_seq = (tent_map(H * W, key + 0.1) * 255).astype(np.uint8).reshape(H, W)
    diffused = np.copy(image)
    for ch in range(C):
        for i in range(0, H, B):
            for j in range(0, W, B):
                block = diffused[i:i+B, j:j+B, ch]
                flat = block.flatten()
                for k in range(1, len(flat)):
                    flat[k] ^= flat[k - 1] ^ block_seq[k]
                block = flat.reshape(block.shape)
                block ^= global_seq[i:i+B, j:j+B]
                diffused[i:i+B, j:j+B, ch] = block
    return diffused

# ------------------ Columnwise Diffusion ------------------
def columnwise_diffuse(image, key):
    H, W, C = image.shape
    seq = (tent_map(H, key + 0.5) * 255).astype(np.uint8)
    diffused = np.copy(image)
    for ch in range(C):
        for j in range(W):
            diffused[:, j, ch] ^= seq
    return diffused

def reverse_columnwise_diffuse(image, key):
    return columnwise_diffuse(image, key)

def reverse_blockwise_diffuse(image, key):
    H, W, C = image.shape
    B = 8
    block_seq = (tent_map(B * B, key) * 255).astype(np.uint8)
    global_seq = (tent_map(H * W, key + 0.1) * 255).astype(np.uint8).reshape(H, W)
    recovered = np.copy(image)
    for ch in range(C):
        for i in range(0, H, B):
            for j in range(0, W, B):
                block = recovered[i:i+B, j:j+B, ch]
                block ^= global_seq[i:i+B, j:j+B]
                flat = block.flatten()
                for k in range(len(flat) - 1, 0, -1):
                    flat[k] ^= flat[k - 1] ^ block_seq[k]
                recovered[i:i+B, j:j+B, ch] = flat.reshape(block.shape)
    return recovered

def reverse_scramble(image, row_perm, col_perm):
    H, W, C = image.shape
    recovered = np.copy(image)
    for ch in range(C):
        recovered[:, :, ch] = recovered[:, np.argsort(col_perm), ch]
        recovered[:, :, ch] = recovered[np.argsort(row_perm), :, ch]
    return recovered

def reverse_preprocess_with_bernoulli(image, key):
    return preprocess_with_bernoulli(image, key)

def reverse_bitplane_xor_with_tent(image, key):
    return bitplane_xor_with_tent(image, key)

# ------------------ Main Encrypt/Decrypt ------------------
def encrypt(image, key):
    img1 = bitplane_xor_with_tent(image, key)
    img2 = preprocess_with_bernoulli(img1, key)
    img3, rp, cp = full_scramble(img2, key)
    img4 = blockwise_diffuse(img3, key)
    encrypted = columnwise_diffuse(img4, key)
    return encrypted, rp, cp

def decrypt(image, key, rp, cp):
    img1 = reverse_columnwise_diffuse(image, key)
    img2 = reverse_blockwise_diffuse(img1, key)
    img3 = reverse_scramble(img2, rp, cp)
    img4 = reverse_preprocess_with_bernoulli(img3, key)
    decrypted = reverse_bitplane_xor_with_tent(img4, key)
    return decrypted

# ------------------ NPCR & UACI ------------------
def calculate_npcr_uaci(original, encrypted):
    diff = original != encrypted
    npcr = np.sum(diff) / diff.size * 100

    uaci = np.sum(np.abs(original.astype(np.int16) - encrypted.astype(np.int16))) / (original.size * 255) * 100

    return npcr, uaci

# ------------------ Streamlit App ------------------
st.title("🔐 RGB Image Encryption using Tent & Bernoulli Maps")

uploaded_file = st.file_uploader("Upload an RGB Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((256, 256))
    image_np = np.array(image)

    key = st.slider("Select Chaos Key (float)", 0.01, 0.99, 0.123456, 0.0001)

    if st.button("Encrypt & Decrypt"):
        start_time = time.time()
        encrypted, rp, cp = encrypt(image_np, key)
        encryption_time = time.time() - start_time

        decrypted = decrypt(encrypted, key, rp, cp)

        npcr, uaci = calculate_npcr_uaci(image_np, encrypted)

        st.subheader("Original Image")
        st.image(image_np, channels="RGB")

        st.subheader("Encrypted Image")
        st.image(encrypted, channels="RGB")

        st.subheader("Decrypted Image")
        st.image(decrypted, channels="RGB")

        st.markdown(f"⏱️ **Time Taken for Encryption:** {encryption_time:.4f} seconds")
        st.markdown(f"🔢 **NPCR:** {npcr:.4f}%")
        st.markdown(f"🔢 **UACI:** {uaci:.4f}%")

