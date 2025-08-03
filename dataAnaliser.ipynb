# ====================================
# 🧠 AI-Driven Data Analysis Notebook
# Gemini + Qwen3-Coder + Supabase + ipywidgets UI
# ====================================

# 🔧 Instalar dependências
!pip install -q google-auth google-auth-oauthlib google-auth-httplib2 \
    gspread pandas matplotlib seaborn ipywidgets requests supabase huggingface_hub

# 🔐 Autenticar Google e montar Drive
from google.colab import auth, drive
auth.authenticate_user()
drive.mount('/content/drive')

# ✅ Tokens de autenticação
import getpass, os
from huggingface_hub import login

# Token HF
hf_token = getpass.getpass("Digite seu Hugging Face token: ")
login(token=hf_token)
os.environ["HF_TOKEN"] = hf_token

# 🔑 Chave Gemini API
gemini_key = getpass.getpass("Digite sua API Key do Gemini (Google AI Studio): ")

# ================================
# 📡 Gemini Flash API – análise direta de dados
# ================================
import requests, json
import pandas as pd

def call_gemini_flash(prompt, df):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
    headers = {"Content-Type": "application/json"}
    
    # Prompt fixo para análise do dataset
    fixed_prompt = f"""
Analise o dataset fornecido e forneça:
1. Principais insights sobre os dados
2. Observações relevantes sobre qualidade/estrutura
3. Padrões interessantes ou anomalias
4. Sugestões de análises complementares

Contexto:
- Total de linhas: {len(df)}
- Total de colunas: {len(df.columns)}
- Colunas: {', '.join(df.columns)}
"""
    
    body = {"contents": [{"parts": [{"text": fixed_prompt}]}]}
    resp = requests.post(url, headers=headers, data=json.dumps(body))
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

# ================================
# 🧠 Qwen3 Coder – geração de código para tarefas
# ================================
from huggingface_hub import InferenceClient

qwen_client = InferenceClient(provider="novita", api_key=os.environ["HF_TOKEN"])

def generate_code(prompt, df):
    cols = ", ".join(df.columns) if df is not None else ""
    code_prompt = f"""
Você é um assistente de programação. Gere apenas código executável em Python para realizar a seguinte tarefa no DataFrame chamado 'df':
Tarefa: {prompt}
Colunas: {cols}

⚠️ Retorne SOMENTE código Python puro e válido, sem markdown, explicações ou comentários. 
Nada além de código e sem usar ``` ou escrever 'python' ou texto explicativo.
"""
    
    response = qwen_client.chat.completions.create(
        model="Qwen/Qwen3-Coder-480B-A35B-Instruct",
        messages=[
            {"role": "user", "content": code_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# ================================
# 🗂️ Google Sheets
# ================================
import gspread
from oauth2client.client import GoogleCredentials

def load_from_gsheet(sheet_url):
    creds = GoogleCredentials.get_application_default()
    client = gspread.authorize(creds)
    sheet = client.open_by_url(sheet_url).sheet1
    return pd.DataFrame(sheet.get_all_records())

# ================================
# 🔗 Supabase
# ================================
from supabase import create_client

supabase_client = None

def setup_supabase(url, key):
    global supabase_client
    supabase_client = create_client(url, key)

def query_supabase(table_name, select="*", filters=None):
    query = supabase_client.table(table_name).select(select)
    if filters:
        for col, op, val in filters:
            query = query.filter(col, op, val)
    response = query.execute()
    return pd.DataFrame(response.data)

# ================================
# 📊 UI: Upload, Google Sheets, Drive, Supabase
# ================================
import ipywidgets as widgets
from IPython.display import display, clear_output
from google.colab import files

df = None
gemini_insights = ""

def handle_data_choice(choice):
    global df, gemini_insights
    if choice == "Upload File":
        uploaded = files.upload()
        for fname in uploaded:
            if fname.endswith(".csv"):
                df = pd.read_csv(fname)
            elif fname.endswith(".xlsx"):
                df = pd.read_excel(fname)
            elif fname.endswith(".json"):
                df = pd.read_json(fname)
            else:
                print("❌ Formato não suportado")
    elif choice == "Google Sheets":
        sheet_url = input("Cole a URL do Google Sheets: ")
        df = load_from_gsheet(sheet_url)
    elif choice == "Supabase":
        table_name = input("Nome da tabela Supabase: ")
        df = query_supabase(table_name)
    elif choice == "Google Drive":
        print("Navegue até /content/drive/MyDrive/... e carregue manualmente o arquivo")
    
    print("\n📄 DataFrame carregado:")
    display(df.head() if df is not None else "Nenhum dado carregado")
    
    # Gerar insights automaticamente com Gemini após carregar dados
    if df is not None and not df.empty:
        print("\n🧠 Gerando insights com Gemini...")
        gemini_insights = call_gemini_flash("", df)
        print("\n📌 Insights Gerais (Gemini):")
        print(gemini_insights)

# UI – escolha da fonte de dados
data_source_selector = widgets.RadioButtons(
    options=['Upload File', 'Google Sheets', 'Supabase', 'Google Drive'],
    description='Data Source:'
)
load_data_btn = widgets.Button(description='Load Data', button_style='primary')

def on_load_data(b):
    clear_output()
    display(gemini_key_input, supabase_url_input, supabase_key_input, setup_btn,
            data_source_selector, load_data_btn, command_input, run_btn, output_box, export_btn)
    handle_data_choice(data_source_selector.value)

load_data_btn.on_click(on_load_data)

# ================================
# 🧪 Execução da Tarefa (somente Qwen3)
# ================================
task_log, code_log, result_log = [], [], []

import traceback

def execute_user_command(cmd):
    global df
    try:
        print(f"\n💡 Task: {cmd}")
        
        # Gerar código com Qwen3
        print("\n🛠️ Gerando código com Qwen3...")
        code = generate_code(cmd, df)
        print("\n📜 Código Gerado:")
        print(code)
        code_log.append(code)

        # Executar código
        local = {'df': df, 'load_from_gsheet': load_from_gsheet, 'query_supabase': query_supabase}
        exec(code, globals(), local)
        if 'df' in local:
            df = local['df']
            print("\n✅ Transformação aplicada com sucesso!")
            print("\n📊 DataFrame atualizado:")
            display(df.head())
        result_log.append("✅ Sucesso")
    except Exception as e:
        print("\n❌ Erro durante execução:")
        traceback.print_exc()
        result_log.append(f"❌ {e}")
    task_log.append(cmd)

# ================================
# 🎛️ Interface completa
# ================================
gemini_key_input = widgets.Text(placeholder='Gemini API Key', description='Gemini Key:')
supabase_url_input = widgets.Text(placeholder='Supabase URL', description='SB URL:')
supabase_key_input = widgets.Password(placeholder='Supabase anon key', description='SB Key:')
setup_btn = widgets.Button(description='Connect Supabase', button_style='warning')

def on_setup_supabase(b):
    setup_supabase(supabase_url_input.value, supabase_key_input.value)
    print("🔗 Supabase conectado!")

setup_btn.on_click(on_setup_supabase)

command_input = widgets.Textarea(
    placeholder='Descreva a tarefa em linguagem natural...',
    description='Comando:',
    layout=widgets.Layout(width='800px', height='80px')
)
run_btn = widgets.Button(description='Executar Tarefa', button_style='success')
output_box = widgets.Output()

def on_run(b):
    output_box.clear_output()
    with output_box:
        execute_user_command(command_input.value)

run_btn.on_click(on_run)

# Exibir UI
display(
    gemini_key_input, supabase_url_input, supabase_key_input, setup_btn,
    data_source_selector, load_data_btn,
    command_input, run_btn, output_box
)

# ================================
# 📤 Exportar relatório
# ================================
from datetime import datetime

export_btn = widgets.Button(description='Exportar Relatório', button_style='info')

def export_report(b):
    df_rep = pd.DataFrame({
        'Tarefa': task_log,
        'Código': code_log,
        'Resultado': result_log,
        'Insights Gerais': [gemini_insights] * len(task_log) if gemini_insights else [""] * len(task_log)
    })
    filename = f"relatorio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_rep.to_csv(filename, index=False)
    print(f"\n📁 Relatório salvo: {filename}")
    files.download(filename)

export_btn.on_click(export_report)
display(export_btn)
