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

2. **Crie um ambiente virtual Python**:
   ```bash
   python -m venv venv
   ```

3. **Ative o ambiente virtual**:
   - **Windows**:
     ```bash
     .\venv\Scripts\activate
     ```
   - **Mac/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Treine o modelo (se necessário)**:
   - Execute o script principal para pré-processar os dados, treinar o modelo e salvar o arquivo `model.pkl`:
     ```bash
     python train_model.py
     ```

6. **Inicie a API Flask**:
   ```bash
   python app.py
   ```

7. **Faça uma requisição para o endpoint**:
   - Envie uma requisição POST para `http://localhost:5000/classify` com o seguinte corpo JSON:
     ```json
     {
       "content": "Texto para análise"
     }
     ```
   - A resposta será:
     ```json
     {
       "classification": "resultado"
     }
     ```

---

## **🗂 Estrutura do Projeto**

```plaintext
.
├── app.py                # Código da API Flask
├── train_model.py        # Código para treinar o modelo
├── requirements.txt      # Dependências do projeto
├── emoji_map.csv         # Mapeamento de emojis
├── giria_map.csv         # Mapeamento de gírias
├── reviews.csv           # Dataset de treinamento
├── model.pkl             # Modelo treinado
├── venv/                 # Ambiente virtual Python
└── README.md             # Documentação do projeto
```

---

## **📊 Resultados**

### Desempenho do Modelo
- **Acurácia média**: 0.85  
- **F1-Score médio**: 0.82  
*Obs.: Métricas calculadas usando validação cruzada com 5 folds.*

---

## **📖 Referências**

- **Bibliotecas usadas**:  
  - [scikit-learn](https://scikit-learn.org)
  - [NLTK](https://www.nltk.org)
  - [Flask](https://flask.palletsprojects.com)

---

## **🤝 Contribuições**

Contribuições são bem-vindas! Sinta-se à vontade para abrir *issues* ou enviar *pull requests*. 

---

## **📝 Licença**

Este projeto está sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.


### O que foi ajustado:

```
- Instruções específicas para o ambiente virtual Python.
- Incluída a pasta `venv` na estrutura do projeto.
- Orientações claras sobre como ativar o ambiente virtual e instalar as dependências.
```
Agora você pode salvar esse conteúdo no arquivo `README.md` no seu repositório do GitHub. 😊











