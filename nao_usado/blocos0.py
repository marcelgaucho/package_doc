# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 13:23:11 2026

@author: Marcel
"""

# %% Import Libraries





# %% Conferência em caso de não convergência do modelo (Massachusetts road dataset)

# Usar depois da validação e antes da finalização das métricas no training loop

# =====================================================================
# BLOCO DE DIAGNÓSTICO DO F1-SCORE (ADAPTADO PARA SOFTMAX)
# =====================================================================
# 1. Captura as probabilidades brutas vindas do Softmax (Shape: 15, 256, 256, 2)
softmax_preds = model(x_v, training=False).numpy()

# 2. Transforma as probabilidades no mapa de classes final (Shape: 15, 256, 256)
# O pixel vira '1' se o canal 1 for maior que o canal 0, caso contrário vira '0'
preds_classes = np.argmax(softmax_preds, axis=-1)

# 3. Converte as labels reais para numpy e garante o formato correto de classe
y_true_np = y_v.numpy() if hasattr(y_v, 'numpy') else np.array(y_v)

# Se o seu gabarito também estiver no formato One-Hot (15, 256, 256, 2), extrai as classes dele também
if len(y_true_np.shape) == 4 and y_true_np.shape[-1] == 2:
    y_true_classes = np.argmax(y_true_np, axis=-1)
else:
    y_true_classes = y_true_np 

# 4. Exibe o Raio-X interpretando os canais do Softmax
print(f"\n" + "="*50)
print(f"[RAIO-X DE DIAGNÓSTICO SOFTMAX | ÉPOCA {epoch+1}]")
print(f" -> Formato original da saída: {softmax_preds.shape}")
print(f" -> Formato após o Argmax (Classes): {preds_classes.shape}")
print(f" -> Confiança média do Canal 0 (Fundo): {softmax_preds[..., 0].mean():.4f}")
print(f" -> Confiança média do Canal 1 (Objeto): {softmax_preds[..., 1].mean():.4f}")
print("-" * 50)
print(f" -> Total de pixels da 'Classe 1' REAIS no lote: {np.sum(y_true_classes == 1)}")
print(f" -> Total de pixels da 'Classe 1' PREVISTOS pelo modelo: {np.sum(preds_classes == 1)}")
print(f" -> Total de pixels da 'Classe 0' PREVISTOS pelo modelo: {np.sum(preds_classes == 0)}")
print("="*50 + "\n")
# =====================================================================