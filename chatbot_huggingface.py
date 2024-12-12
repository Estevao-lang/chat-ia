import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify, render_template

# Inicialize o modelo e o tokenizer do DialoGPT
model_name = "microsoft/DialoGPT-large"  # Usando DialoGPT para chat
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define o token de padding
tokenizer.pad_token = tokenizer.eos_token  # Usar o token EOS como token de padding
pad_token_id = tokenizer.pad_token_id

# Configura o dispositivo (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Inicializa o aplicativo Flask
app = Flask(__name__)

# Função para gerar a resposta
def gerar_resposta(input_text):
    try:
        # Codifica a entrada com o tokenizer
        input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt").to(device)
        attention_mask = input_ids.ne(pad_token_id)

        # Gera a resposta com o modelo DialoGPT
        response_ids = model.generate(
            input_ids,
            max_length=200,  # Limite de comprimento da resposta
            pad_token_id=pad_token_id,
            attention_mask=attention_mask,
            do_sample=True,  # Amostragem para gerar uma resposta mais variada
            temperature=0.7,  # Controle de aleatoriedade
            top_p=0.9         # Controle de diversidade de palavras
        )
        # Decodifica a resposta gerada para o formato de texto
        response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"Erro ao gerar resposta: {e}")
        return "Desculpe, houve um erro ao gerar a resposta."

# Rota principal para renderizar a página inicial
@app.route('/')
def index():
    return render_template('index.html')

# Rota para a página de chat (GET)
@app.route('/chat', methods=['GET'])
def chat():
    return render_template('chat.html')

# Rota para interagir com o chatbot via API (POST)
@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.json
    user_input = data.get("input_text")
    
    if not user_input:
        return jsonify({"error": "Texto de entrada ausente"}), 400
    
    resposta = gerar_resposta(user_input)
    return jsonify({"response": resposta})

# Inicializa o servidor
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
