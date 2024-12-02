# **Classificação de Sentimentos com Regressão Logística**

Este projeto implementa um sistema de classificação de sentimentos utilizando Regressão Logística e um pipeline de pré-processamento de texto personalizado. Ele inclui um modelo treinado, uma API para classificação de texto e exemplos de uso.

---

## **📋 Funcionalidades**

- **Pré-processamento de texto**:  
  - Substituição de emojis e gírias por equivalentes textuais.
  - Normalização de caracteres repetidos e elementos não textuais.
  - Remoção de *stopwords* e aplicação de *stemming*.
  - Vetorização manual com *Bag of Words*.

- **Classificação de textos**:
  - Utiliza Regressão Logística para classificação de sentimentos.
  - Modelo treinado usando validação cruzada para avaliar desempenho.

- **API REST**:  
  - Endpoint `/classify` para realizar a classificação de novos textos via HTTP.

---

## **🚀 Como executar**

### **Pré-requisitos**
- Python 3.8 ou superior.
- Ambiente virtual Python configurado.
- Bibliotecas necessárias (instaláveis via `requirements.txt`).
- Arquivos de suporte:  
  - `emoji_map.csv`: Mapeamento de emojis para sentimentos.  
  - `giria_map.csv`: Mapeamento de gírias para termos formais.  
  - `reviews.csv`: Dataset de treinamento (reviews e classificações).

### **Passos para execução**

1. **Clone o repositório**:
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio
