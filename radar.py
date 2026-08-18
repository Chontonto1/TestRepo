import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import time
import warnings
import threading
import asyncio
import math
import json
import ssl
import urllib.request
import csv
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta, timezone
from telegram import Bot, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler
from stable_baselines3 import PPO

# --- CONFIGURACIÓN TELEGRAM ---
TELEGRAM_TOKEN = "8431200001:AAHaHphT53lI5n8kXsS5cqboFUba4WOFqok"
TELEGRAM_CHAT_ID = "8521648172"

# --- CREDENCIALES MT5 ---
MT5_LOGIN = 52715912 
MT5_PASSWORD = "tCR22xiE$8b5zP" 
MT5_SERVER = "ICMarketsSC-Demo"

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

# --- ARQUITECTURA DEL MODELO MODIFICADA PARA CLASIFICACIÓN MULTICLASE ---
class TradingLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=3):
        super(TradingLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3) # Salida de 3 clases: 0=Neutro, 1=Buy, 2=Sell
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- VARIABLES GLOBALES DINÁMICAS (CONFIGURADAS POR EL LANZADOR) ---
PARES = []
PARES_ACTIVOS = {} # Diccionario de control de pausa en tiempo real
TIMEFRAME_SELECCIONADO = mt5.TIMEFRAME_M30
TIMEFRAME_NOMBRE = "M30"
LOTE = 0.01
BALANCE_SIMULADO = 300.0

# --- MAPA DE TIMEFRAMES MT5 ---
MAPA_TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1
}

modelos_lstm = {}
modelos_ppo = {}
scalers_x = {}
scalers_y = {}
estado_radar = {}
posiciones_activas = {}
ultima_vela_procesada = {}

# --- CACHE DE FILTRO DE NOTICIAS ---
ULTIMA_CONSULTA_NOTICIAS = None
NOTICIAS_ACTIVAS_DIVISAS = set()

# --- CONFIGURACIÓN OPTIMIZADA DE REGLAS SEGÚN METRICAS DEL BACKTEST ---
REGLAS_PAR = {
    'EURUSD': {
        'use_trailing': False,     
        'use_macro_filter': False  
    },
    'USDCHF': {
        'use_trailing': False,      
        'use_macro_filter': False   
    },
    'USDCAD': {
        'use_trailing': False,      
        'use_macro_filter': False   
    }
}

def registrar_log(mensaje):
    fecha_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linea = f"[{fecha_str}] {mensaje}"
    with open("registro_radar.log", "a", encoding="utf-8") as f:
        f.write(linea + "\n")

def registrar_operacion_csv(simbolo, evento, tipo="", ticket=0, precio=0.0, sl=0.0, pnl=0.0, motivo="", rsi=0.0, adx=0.0):
    archivo_csv = "registro_operaciones.csv"
    file_exists = os.path.isfile(archivo_csv)
    
    with open(archivo_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Simbolo", "Evento", "Tipo", "Ticket", "Precio", "SL", "PnL", "Motivo", "RSI", "ADX"])
        
        fecha_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([fecha_str, simbolo, evento, tipo, ticket, f"{precio:.5f}", f"{sl:.5f}", f"{pnl:.2f}", motivo, f"{rsi:.1f}", f"{adx:.1f}"])

def enviar_telegram_sync(mensaje):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        bot = Bot(token=TELEGRAM_TOKEN.strip())
        loop.run_until_complete(bot.send_message(chat_id=TELEGRAM_CHAT_ID.strip(), text=mensaje))
        loop.close()
    except Exception as e:
        pass

# --- MÓDULO DE FILTRO DE NOTICIAS MACRO MEJORADO Y OPTIMIZADO ---
def obtener_noticias_alto_impacto():
    global ULTIMA_CONSULTA_NOTICIAS, NOTICIAS_ACTIVAS_DIVISAS
    ahora = datetime.now(timezone.utc)
    
    if ULTIMA_CONSULTA_NOTICIAS is not None and (ahora - ULTIMA_CONSULTA_NOTICIAS).total_seconds() < 600:
        return NOTICIAS_ACTIVAS_DIVISAS

    url = "https://s3.amazonaws.com/forexfactory.apps/calendar/thisWeek.json"
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(
        url, 
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json'
        }
    )
    
    divisas_afectadas = set()
    try:
        with urllib.request.urlopen(req, timeout=4, context=ctx) as response:
            data = json.loads(response.read().decode('utf-8'))
            for evento in data:
                impacto = str(evento.get('impact', '')).strip().lower()
                if impacto in ['high', 'alto', '3']:
                    fecha_str = evento.get('date', '')
                    if not fecha_str:
                        continue
                    
                    try:
                        fecha_evento = datetime.fromisoformat(fecha_str.replace('Z', '+00:00'))
                    except ValueError:
                        continue

                    inicio_bloqueo = fecha_evento - timedelta(minutes=30)
                    fin_bloqueo = fecha_evento + timedelta(minutes=15)
                    
                    if inicio_bloqueo <= ahora <= fin_bloqueo:
                        currency = str(evento.get('country', '')).upper().strip()
                        if currency:
                            divisas_afectadas.add(currency)

        ULTIMA_CONSULTA_NOTICIAS = ahora
        NOTICIAS_ACTIVAS_DIVISAS = divisas_afectadas
    except Exception as e:
        ULTIMA_CONSULTA_NOTICIAS = ahora  
        
    return NOTICIAS_ACTIVAS_DIVISAS

def par_afectado_por_noticia(simbolo):
    divisas_en_alerta = obtener_noticias_alto_impacto()
    if not divisas_en_alerta:
        return False
    base = simbolo[:3]
    quote = simbolo[3:]
    return (base in divisas_en_alerta) or (quote in divisas_en_alerta)

# --- GUI LANZADOR ---
def mostrar_lanzador():
    global PARES, PARES_ACTIVOS, TIMEFRAME_SELECCIONADO, TIMEFRAME_NOMBRE, LOTE
    
    root = tk.Tk()
    root.title("Lanzador de Configuración - Radar Trading")
    root.geometry("380x420")
    root.resizable(False, False)
    
    pares_disponibles = ['EURUSD', 'USDCHF', 'USDCAD']
    var_pares = {}
    
    tk.Label(root, text="⚙️ CONFIGURACIÓN DEL RADAR", font=("Arial", 12, "bold")).pack(pady=10)
    
    frame_pares = tk.LabelFrame(root, text=" Seleccionar Activos (Marcados = Activos al inicio) ", font=("Arial", 8, "bold"))
    frame_pares.pack(fill="x", padx=15, pady=5)
    for p in pares_disponibles:
        var = tk.BooleanVar(value=True)
        chk = tk.Checkbutton(frame_pares, text=p, variable=var, font=("Arial", 10))
        chk.pack(anchor="w", padx=10, pady=2)
        var_pares[p] = var

    frame_tf = tk.LabelFrame(root, text=" Temporabilidad (Timeframe) ", font=("Arial", 9, "bold"))
    frame_tf.pack(fill="x", padx=15, pady=5)
    combo_tf = ttk.Combobox(frame_tf, values=list(MAPA_TIMEFRAMES.keys()), state="readonly", font=("Arial", 10))
    combo_tf.set("M30")
    combo_tf.pack(padx=10, pady=8, fill="x")

    frame_lote = tk.LabelFrame(root, text=" Gestión de Lotaje ", font=("Arial", 9, "bold"))
    frame_lote.pack(fill="x", padx=15, pady=5)
    entry_lote = tk.Entry(frame_lote, font=("Arial", 10))
    entry_lote.insert(0, "0.01")
    entry_lote.pack(padx=10, pady=8, fill="x")

    def iniciar_sistema():
        global PARES, PARES_ACTIVOS, TIMEFRAME_SELECCIONADO, TIMEFRAME_NOMBRE, LOTE, posiciones_activas, ultima_vela_procesada
        
        PARES = list(pares_disponibles)
        
        try:
            val_lote = float(entry_lote.get())
            if val_lote <= 0:
                raise ValueError
            LOTE = val_lote
        except ValueError:
            messagebox.showerror("Error", "Ingrese un lotaje válido mayor a 0.")
            return

        TIMEFRAME_NOMBRE = combo_tf.get()
        TIMEFRAME_SELECCIONADO = MAPA_TIMEFRAMES[TIMEFRAME_NOMBRE]

        PARES_ACTIVOS = {p: var_pares[p].get() for p in PARES}
        posiciones_activas = {p: None for p in PARES}
        ultima_vela_procesada = {p: None for p in PARES}

        root.destroy()

    btn_iniciar = tk.Button(root, text="🚀 INICIAR RADAR", command=iniciar_sistema, bg="#2196F3", fg="white", font=("Arial", 11, "bold"), height=2)
    btn_iniciar.pack(fill="x", padx=15, pady=15)

    root.mainloop()

# --- COMANDOS TELEGRAM ---
async def cmd_status(update, context):
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        await update.message.reply_text("❌ MT5 Desconectado.")
        return
        
    msg = "📊 ESTADO DE LOS ACTIVOS VIGENTES:\n"
    msg += "─" * 25 + "\n"
    
    for p in PARES:
        datos = estado_radar.get(p, {"status": "Iniciando...", "adx": 0.0})
        status_actual = datos["status"]
        
        if "DENTRO" in status_actual or "COMPRA" in status_actual or "VENTA" in status_actual:
            posiciones = mt5.positions_get(symbol=p)
            if posiciones:
                pnl_par = sum(pos.profit for pos in posiciones)
                status_actual += f" (PnL: ${pnl_par:.2f})"
            else:
                status_actual += " (PnL: $0.00)"
                
        msg += f"💱 {p}: {status_actual}\n"
        
    await update.message.reply_text(msg)

async def cmd_radar(update, context):
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        await update.message.reply_text("❌ MT5 Desconectado.")
        return
        
    account_info = mt5.account_info()
    if account_info is None:
        await update.message.reply_text("❌ No se pudo obtener la información de la cuenta.")
        return
        
    posiciones = mt5.positions_get()
    pnl_total_flotante = sum(pos.profit for pos in posiciones if pos.symbol in PARES) if posiciones else 0.0
    
    msg = "🚀 METRICAS CONSOLIDADAS DEL RADAR:\n"
    msg += "─" * 30 + "\n"
    msg += f"💰 Balance Cuenta: ${account_info.balance:.2f}\n"
    msg += f"📈 Equidad Actual: ${account_info.equity:.2f}\n"
    msg += f"📊 PnL Flotante Total: ${pnl_total_flotante:.2f}\n"
    msg += f"🔄 Margen Libre: ${account_info.margin_free:.2f}\n"
    msg += "─" * 30 + "\n"
    msg += f"🤖 Sistema Operando en Tiempo Real ({', '.join(PARES)}) | TF: {TIMEFRAME_NOMBRE} | Lotaje: {LOTE}."
    
    await update.message.reply_text(msg)

async def cmd_pause(update, context):
    if not context.args:
        keyboard = [[InlineKeyboardButton(p, callback_data=f"pause_{p}")] for p in PARES]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("⏸️ Selecciona el par que deseas PAUSAR:", reply_markup=reply_markup)
        return
    simbolo = context.args[0].upper()
    if simbolo in PARES:
        PARES_ACTIVOS[simbolo] = False
        if simbolo in estado_radar:
            estado_radar[simbolo]['ppo_status'] = "Pausado"
        registrar_log(f"⏸️ CONTROL TELEGRAM: Se pausó el monitoreo del par {simbolo}.")
        await update.message.reply_text(f"⏸️ El par {simbolo} ha sido PAUSADO correctamente.")
    else:
        await update.message.reply_text(f"❌ El par {simbolo} no forma parte de los pares cargados en el radar.")

async def cmd_resume(update, context):
    if not context.args:
        keyboard = [[InlineKeyboardButton(p, callback_data=f"resume_{p}")] for p in PARES]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("▶️ Selecciona el par que deseas REANUDAR:", reply_markup=reply_markup)
        return
    simbolo = context.args[0].upper()
    if simbolo in PARES:
        PARES_ACTIVOS[simbolo] = True
        if simbolo in estado_radar:
            estado_radar[simbolo]['ppo_status'] = "Listo"
        registrar_log(f"▶️ CONTROL TELEGRAM: Se reanudó el monitoreo del par {simbolo}.")
        await update.message.reply_text(f"▶️ El par {simbolo} ha sido REACTIVADO correctamente.")
    else:
        await update.message.reply_text(f"❌ El par {simbolo} no forma parte de los pares cargados en el radar.")

async def callback_button_handler(update, context):
    query = update.callback_query
    await query.answer()
    
    data = query.data
    if data.startswith("pause_"):
        simbolo = data.replace("pause_", "")
        if simbolo in PARES:
            PARES_ACTIVOS[simbolo] = False
            if simbolo in estado_radar:
                estado_radar[simbolo]['ppo_status'] = "Pausado"
            registrar_log(f"⏸️ CONTROL TELEGRAM: Se pausó el monitoreo del par {simbolo}.")
            await query.edit_message_text(f"⏸️ El par {simbolo} ha sido PAUSADO correctamente.")
    elif data.startswith("resume_"):
        simbolo = data.replace("resume_", "")
        if simbolo in PARES:
            PARES_ACTIVOS[simbolo] = True
            if simbolo in estado_radar:
                estado_radar[simbolo]['ppo_status'] = "Listo"
            registrar_log(f"▶️ CONTROL TELEGRAM: Se reanudó el monitoreo del par {simbolo}.")
            await query.edit_message_text(f"▶️ El par {simbolo} ha sido REACTIVADO correctamente.")

async def cmd_kill_system(update, context):
    await update.message.reply_text("🚨 ORDEN CRÍTICA RECIBIDA: Cerrando operaciones y apagando el sistema...")
    registrar_log("🚨 COMANDO CRÍTICO: Executing Kill_System. Cerrando posiciones y apagando PC.")
    
    if mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        posiciones = mt5.positions_get()
        if posiciones:
            for pos in posiciones:
                if pos.symbol in PARES:
                    ticket = pos.ticket
                    symbol = pos.symbol
                    tipo_orden = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                    precio = mt5.symbol_info_tick(symbol).bid if tipo_orden == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(symbol).ask
                    
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": pos.volume,
                        "type": tipo_orden,
                        "position": ticket,
                        "price": precio,
                        "deviation": 20,
                        "magic": 234567,
                        "comment": "Kill System Closure",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    mt5.order_send(request)
        mt5.shutdown()
        
    await update.message.reply_text("✅ Operaciones cerradas. Ejecutando apagado de hardware...")
    os.system("shutdown /s /t 1")

async def configurar_menu_comandos(application: Application):
    try:
        await application.bot.delete_my_commands()
        comandos = [
            BotCommand("radar", "Métricas de la cuenta, balance y PnL"),
            BotCommand("status", "Estado detallado de los pares en vivo"),
            BotCommand("pause", "Pausar monitoreo de un activo"),
            BotCommand("resume", "Reanudar monitoreo de un activo"),
            BotCommand("kill_system", "Cierra todo y apaga la PC")
        ]
        await application.bot.set_my_commands(comandos)
    except Exception as e:
        registrar_log(f"⚠️ No se pudo actualizar el menú de comandos en Telegram: {e}")

# --- FUNCIONES DE MERCADO E INFRAESTRUCTURA ---
def cargar_infraestructura():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registrar_log(f"🧠 Inicializando Tensores en hardware dedicado: {device}")
    
    for p in PARES:
        pl = p.lower()
        try:
            scalers_x[p] = joblib.load(f"scalers/scaler_x_{pl}.pkl")
            scalers_y[p] = joblib.load(f"scalers/scaler_y_{pl}.pkl")
            
            modelos_lstm[p] = TradingLSTM().to(device)
            modelos_lstm[p].load_state_dict(torch.load(f"models/model_{pl}.pth", map_location=device))
            modelos_lstm[p].eval()
            
            modelos_ppo[p] = PPO.load(f"agente_ppo_{pl}.zip", device=device)
            
            ppo_init_status = "Listo" if PARES_ACTIVOS.get(p, True) else "Pausado"
            estado_radar[p] = {"status": "Modelo Cargado", "adx": 0.0, "ppo_status": ppo_init_status, "ppo": modelos_ppo[p]}
        except Exception as e:
            registrar_log(f"❌ Fallo crítico cargando infraestructura para {p}: {e}")

def calcular_indicadores(df):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    df['macd_hist'] = macd_line - macd_line.ewm(span=9, adjust=False).mean()
    
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / (df['atr'] + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / (df['atr'] + 1e-9))
    df['adx'] = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)).ewm(alpha=1/14, adjust=False).mean()
    
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean() 
    df['volat'] = df['close'].rolling(window=20).std()
    df['t_vol'] = df['tick_volume'].astype(float)
    return df

def obtener_datos_par(simbolo):
    rates = mt5.copy_rates_from_pos(simbolo, TIMEFRAME_SELECCIONADO, 0, 1000)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = calcular_indicadores(df)
    return df.dropna().reset_index(drop=True)

def verificar_cierres_historico():
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        return
    ahora = datetime.now()
    desde = ahora - timedelta(hours=4)
    historico = mt5.history_deals_get(desde, ahora)
    if historico:
        for deal in historico:
            if deal.entry == 1: # DEAL_ENTRY_OUT
                ticket_posicion = deal.position_id
                for p in PARES:
                    if posiciones_activas.get(p) == ticket_posicion:
                        reason = deal.reason
                        if reason == getattr(mt5, 'DEAL_REASON_SL', 3):
                            motivo = "Stop Loss (SL) alcanzado"
                        elif reason == getattr(mt5, 'DEAL_REASON_TP', 4):
                            motivo = "Take Profit (TP) alcanzado"
                        elif reason == getattr(mt5, 'DEAL_REASON_SO', 5):
                            motivo = "Stop Out (SO)"
                        elif reason == getattr(mt5, 'DEAL_REASON_EXPERT', 1):
                            motivo = "Cierre por Algoritmo/Experto"
                        else:
                            motivo = f"Cierre Externo (Reason: {reason})"

                        tipo_str = "BUY" if deal.type == mt5.DEAL_TYPE_SELL else "SELL"
                        registrar_log(f"[{p}] POSICIÓN CERRADA EXTERNAMENTE DETECTADA -> Ticket: {ticket_posicion} | Motivo: {motivo} | PnL: ${deal.profit:.2f}")
                        registrar_operacion_csv(p, "CIERRE_EXTERNO", tipo_str, ticket_posicion, deal.price, 0.0, deal.profit, motivo, 0.0, 0.0)
                        enviar_telegram_sync(f"🚨 [{p}] POSICIÓN CERRADA (Servidor/SL)\nMotivo: {motivo}\nTicket: {ticket_posicion}\nPnL: ${deal.profit:.2f}")
                        posiciones_activas[p] = None

def ejecutar_cierre_mercado(simbolo, posicion, motivo="", rsi=0.0, adx=0.0):
    ticket = posicion.ticket
    tipo_orden_cierre = mt5.ORDER_TYPE_SELL if posicion.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tipo_str = "BUY" if posicion.type == mt5.ORDER_TYPE_BUY else "SELL"
    pnl_actual = posicion.profit
    
    max_intentos = 5
    for intento in range(1, max_intentos + 1):
        tick_actual = mt5.symbol_info_tick(simbolo)
        if tick_actual is None:
            registrar_log(f"❌ [{simbolo}] Intento {intento}/{max_intentos}: Error obteniendo ticks del mercado para el precio de cierre.")
            time.sleep(1.5)
            continue
            
        precio = tick_actual.bid if tipo_orden_cierre == mt5.ORDER_TYPE_SELL else tick_actual.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": simbolo,
            "volume": float(posicion.volume),
            "type": tipo_orden_cierre,
            "position": int(ticket),
            "price": float(precio),
            "deviation": 30,
            "magic": 234567,
            "comment": "Test Close Fixed",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        resultado = mt5.order_send(request)
        
        if resultado is None:
            err_code, err_str = mt5.last_error()
            registrar_log(f"⚠️ [{simbolo}] Intento {intento}/{max_intentos}: order_send devolvió None. MT5 Error: {err_code} ({err_str}). Reintentando...")
            time.sleep(1.5)
            continue

        if resultado.retcode == mt5.TRADE_RETCODE_DONE:
            posiciones_activas[simbolo] = None
            registrar_log(f"🚨 [{simbolo}] POSICIÓN CERRADA DE FORMA AUTOMÁTICA. Motivo: {motivo}. Procesado en intento {intento}.")
            registrar_operacion_csv(simbolo, "CIERRE", tipo_str, ticket, precio, posicion.sl, pnl_actual, motivo, rsi, adx)
            enviar_telegram_sync(f"🚨 [{simbolo}] POSICIÓN CERRADA\nMotivo: {motivo}\nTicket: {ticket}\nPrecio Cierre: {precio}")
            return
        else:
            registrar_log(f"❌ [{simbolo}] Fallo al ejecutar orden de cierre: {resultado.comment} (Código Retcode: {resultado.retcode})")
            time.sleep(1.5)
            
    registrar_log(f"🚨 [CRÍTICO] [{simbolo}] Se agotaron los {max_intentos} intentos de cierre. La operación permanece abierta en el terminal.")

def gestionar_trailing_stop(simbolo, posicion, info_simbolo, tick_actual, atr_valor, rsi=0.0, adx=0.0):
    if not REGLAS_PAR.get(simbolo, {}).get('use_trailing', False):
        return

    point = info_simbolo.point
    digits = info_simbolo.digits
    
    sl_actual = float(posicion.sl)
    precio_entrada = float(posicion.price_open)
    ticket = posicion.ticket
    tipo_str = "BUY" if posicion.type == mt5.ORDER_TYPE_BUY else "SELL"
    
    if simbolo == 'USDCHF':
        ts_activacion = 2.5  
        ts_distancia = 2.0
    elif simbolo == 'USDCAD':
        ts_activacion = 3.0
        ts_distancia = 2.0
    else:
        ts_activacion = 3.5
        ts_distancia = 2.5

    activacion_puntos = ts_activacion * atr_valor
    distancia_puntos = ts_distancia * atr_valor
    
    nuevo_sl = 0.0
    
    if posicion.type == mt5.ORDER_TYPE_BUY:
        precio_actual = float(tick_actual.bid)
        if (precio_actual - precio_entrada) >= activacion_puntos:
            target_sl = precio_actual - distancia_puntos
            if target_sl > sl_actual:
                nuevo_sl = round(target_sl, digits)
                
    elif posicion.type == mt5.ORDER_TYPE_SELL:
        precio_actual = float(tick_actual.ask)
        if (precio_entrada - precio_actual) >= activacion_puntos:
            target_sl = precio_actual + distancia_puntos
            if sl_actual == 0.0 or target_sl < sl_actual:
                nuevo_sl = round(target_sl, digits)

    if nuevo_sl > 0.0 and nuevo_sl != sl_actual:
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(ticket),
            "symbol": simbolo,
            "sl": nuevo_sl,
            "tp": float(posicion.tp)
        }
        resultado = mt5.order_send(request)
        if resultado and resultado.retcode == mt5.TRADE_RETCODE_DONE:
            registrar_log(f"🎯 [{simbolo}] TRAILING STOP ADAPTATIVO ACTUALIZADO -> Ticket: {ticket} | Nuevo SL: {nuevo_sl:.5f}")
            registrar_operacion_csv(simbolo, "TRAILING_STOP", tipo_str, ticket, tick_actual.bid, nuevo_sl, posicion.profit, "Ajuste Trailing Stop", rsi, adx)
            enviar_telegram_sync(f"🎯 [{simbolo}] TRAILING STOP DINÁMICO\nTicket: {ticket}\nNuevo SL Modificado: {nuevo_sl:.5f}")
        else:
            err_msg = resultado.comment if resultado else "order_send nulo"
            registrar_log(f"⚠️ [{simbolo}] Error modificando Trailing Stop: {err_msg}")

def generar_vector_observacion(df_par, pred_class, simbolo):
    posiciones = mt5.positions_get(symbol=simbolo)
    pnl_flotante_par = sum(pos.profit for pos in posiciones) if posiciones else 0.0
    
    equity_simulada = max(0.0, BALANCE_SIMULADO + pnl_flotante_par)
    equity_relativa = equity_simulada / BALANCE_SIMULADO
    drawdown_actual = max(0.0, (BALANCE_SIMULADO - equity_simulada) / BALANCE_SIMULADO)

    fila = df_par.iloc[-1]
    lstm_0 = 1.0 if pred_class == 0 else 0.0
    lstm_1 = 1.0 if pred_class == 1 else 0.0
    lstm_2 = 1.0 if pred_class == 2 else 0.0

    features = [
        float(fila['close']), float(fila['rsi']), float(fila['macd_hist']),
        float(fila['adx']), float(fila['ema_20']), float(fila['volat']),
        float(fila['t_vol']), lstm_0, lstm_1, lstm_2,
        float(drawdown_actual), float(equity_relativa)
    ]

    agente = modelos_ppo.get(simbolo)
    if agente is not None:
        expected_shape = agente.observation_space.shape[0]
        if expected_shape == 13:
            ema_200 = float(fila['ema_200']) if 'ema_200' in fila else float(fila['close'])
            tendencia_macro = 1.0 if float(fila['close']) >= ema_200 else -1.0
            features.append(tendencia_macro)

    return np.array(features, dtype=np.float32)

def enviar_orden(simbolo, tipo_orden, pred_class, agente_ppo, df_par, action):
    tipo_sugerido = "COMPRA" if tipo_orden == mt5.ORDER_TYPE_BUY else "VENTA"
    tipo_str = "BUY" if tipo_orden == mt5.ORDER_TYPE_BUY else "SELL"
    
    rsi_val = float(df_par.iloc[-1]['rsi'])
    adx_val = float(df_par.iloc[-1]['adx'])
    
    registrar_log(f"[{simbolo}] Evaluando sugerencia de {tipo_sugerido}. Acción dictaminada por PPO: {action}")
    
    accion_valida = (pred_class == 1 and action == 1) or (pred_class == 2 and action == 2)
    
    if not accion_valida:
        dictamen_str = "Omitir (0)" if action == 0 else ("Compra (1)" if action == 1 else ("Venta (2)" if action == 2 else "Cerrar (3)"))
        registrar_log(f"[{simbolo}] VETO PPO: {tipo_sugerido} rechazada. El agente decidió dictaminar: {dictamen_str}.")
        registrar_operacion_csv(simbolo, "VETO", tipo_str, motivo=f"VETO PPO ({dictamen_str})", rsi=rsi_val, adx=adx_val)
        estado_radar[simbolo]['ppo_status'] = f"VETO ({dictamen_str})"
        return None

    registrar_log(f"[{simbolo}] VALIDADO PPO: El modelo RL aprueba la entrada al mercado.")
    estado_radar[simbolo]['ppo_status'] = "APROBADO"

    tick = mt5.symbol_info_tick(simbolo)
    info_simbolo = mt5.symbol_info(simbolo)
    if tick is None or info_simbolo is None:
        registrar_log(f"❌ [{simbolo}] Error al obtener ticks o info del mercado.")
        return None
        
    precio = tick.ask if tipo_orden == mt5.ORDER_TYPE_BUY else tick.bid
    point = info_simbolo.point
    
    tick_value = info_simbolo.trade_tick_value
    tick_size = info_simbolo.trade_tick_size
    puntos_comision = ((0.70 / (LOTE * tick_value)) * tick_size) / point if tick_value > 0 else 20.0
    
    spread_puntos = info_simbolo.spread
    puntos_totales_cobertura = int(math.ceil(spread_puntos + puntos_comision)) + 10 
    
    atr_valor = float(df_par.iloc[-1]['atr']) if 'atr' in df_par.columns and float(df_par.iloc[-1]['atr']) > 0 else (30 * point)
    distancia_sl = max(puntos_totales_cobertura * point, 1.5 * atr_valor)
    
    if tipo_orden == mt5.ORDER_TYPE_BUY:
        sl_precio = precio - distancia_sl
    else:
        sl_precio = precio + distancia_sl
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": simbolo,
        "volume": float(LOTE),
        "type": int(tipo_orden),
        "price": float(precio),
        "sl": round(float(sl_precio), info_simbolo.digits),
        "deviation": 20,
        "magic": 234567,
        "comment": "Simulacion Real 300USD",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    resultado = mt5.order_send(request)
    
    if resultado is None:
        err_code, err_str = mt5.last_error()
        registrar_log(f"❌ [{simbolo}] Fallo estructural: order_send devolvió None al lanzar orden. MT5 Error: {err_code} ({err_str})")
        return None
        
    if resultado.retcode == mt5.TRADE_RETCODE_DONE:
        ticket_valido = resultado.order if resultado.order != 0 else resultado.deal
            
        time.sleep(0.1)
        posiciones_reales = mt5.positions_get(symbol=simbolo)
        if posiciones_reales:
            ticket_valido = posiciones_reales[0].ticket
            
        posiciones_activas[simbolo] = ticket_valido
        registrar_log(f"[{simbolo}] ORDEN EJECUTADA (SIMULACIÓN REAL) -> Ticket Position: {ticket_valido} | Lote: {LOTE} | Precio: {precio} | SL: {sl_precio:.5f}")
        registrar_operacion_csv(simbolo, "ENTRADA", tipo_str, ticket_valido, precio, sl_precio, 0.0, "Entrada Validada PPO", rsi_val, adx_val)
        enviar_telegram_sync(f"🚀 [{simbolo}] ORDEN EJECUTADA (SIMULACIÓN REAL)\nTipo: {tipo_sugerido}\nLote: {LOTE}\nPrecio: {precio}\nSL (Ajustado comisiones): {sl_precio:.5f}\nTicket Asignado: {ticket_valido}")
    else:
        registrar_log(f"❌ [{simbolo}] Fallo estructural al lanzar orden en MT5: {resultado.comment} (Código: {resultado.retcode})")
    return resultado

def bucle_radar():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    while True:
        if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            time.sleep(5)
            continue
            
        verificar_cierres_historico()
        ahora = datetime.now()
        
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("═"*70)
        print(f"🛰️  MONITOR RADAR ({TIMEFRAME_NOMBRE}) | LOTE: {LOTE} | BALANCE SIMULADO: ${BALANCE_SIMULADO}")
        print(f"⏰ Hora Local: {ahora.strftime('%Y-%m-%d %H:%M:%S')}")
        print("═"*70)
        print(f"{'ACTIVO':<12} | {'ADX':<9} | {'CLASE LSTM':<11} | {'FILTRO PPO':<13} | {'ESTADO RADAR'}")
        print("─"*70)
        
        for s in PARES:
            data = estado_radar[s] if s in estado_radar else {"status": "Iniciando...", "adx": 0.0, "ppo_status": "Listo", "ppo": modelos_ppo.get(s)}
            
            if not PARES_ACTIVOS.get(s, True):
                status = "⏸️ PAUSADO"
                adx_now, pred_class = 0.0, 0
                estado_radar[s] = {"status": status, "adx": adx_now, "ppo_status": "Pausado", "ppo": data.get('ppo')}
                adx_str = f"ADX: {adx_now:.1f}"
                pred_str = f"CLASS: {pred_class}"
                ppo_str = f"[PPO: Pausado]"
                print(f"{s:<12} | {adx_str:<9} | {pred_str:<11} | {ppo_str:<13} | {status}")
                continue

            current_ppo_status = data.get('ppo_status', 'Listo')
            if current_ppo_status == "Pausado":
                current_ppo_status = "Listo"

            posiciones = mt5.positions_get(symbol=s)
            info_simbolo = mt5.symbol_info(s)
            tick_actual = mt5.symbol_info_tick(s)
            
            df = obtener_datos_par(s)
            if df is None or len(df) < 61 or info_simbolo is None or tick_actual is None:
                adx_now, pred_class = 0.0, 0
                status = "❌ SIN DATOS"
            else:
                ultimo_row = df.iloc[-2] 
                timestamp_vela = ultimo_row['time']
                adx_now = float(ultimo_row['adx'])
                rsi_now = float(ultimo_row['rsi'])
                atr_valor = float(ultimo_row['atr'])
                
                close_actual = float(ultimo_row['close'])
                ema_200_actual = float(ultimo_row['ema_200'])
                
                features_cols = ['close', 'rsi', 'macd_hist', 'adx', 'ema_20', 'volat', 't_vol']
                ventana = df.iloc[-61:-1][features_cols] 
                ventana_scaled = scalers_x[s].transform(ventana)
                
                with torch.no_grad():
                    X_tensor = torch.tensor(ventana_scaled, dtype=torch.float32).unsqueeze(0).to(device)
                    outputs = modelos_lstm[s](X_tensor)
                    _, preds = torch.max(outputs, 1)
                    pred_class = int(preds.cpu().item())
                
                if posiciones:
                    status = f"🟩 DENTRO ({'BUY' if posiciones[0].type==0 else 'SELL'})"
                    posiciones_activas[s] = posiciones[0].ticket
                    
                    gestionar_trailing_stop(s, posiciones[0], info_simbolo, tick_actual, atr_valor, rsi_now, adx_now)
                    
                    # Conteo de velas transcurridas desde la apertura de la posición
                    pos_time = pd.to_datetime(posiciones[0].time, unit='s')
                    velas_en_posicion = (df['time'] > pos_time).sum()

                    # Regla de permanencia máxima (30 velas para USDCAD, USDCHF, EURGBP)
                    cierre_por_tiempo = False
                    if s in ['USDCAD', 'USDCHF', 'EURGBP'] and velas_en_posicion >= 30:
                        ejecutar_cierre_mercado(s, posiciones[0], "Límite 30 velas alcanzado", rsi_now, adx_now)
                        status = "🚨 CIERRE LÍMITE VELAS"
                        cierre_por_tiempo = True

                    if not cierre_por_tiempo:
                        obs_vector = generar_vector_observacion(df.iloc[:-1], pred_class, s)
                        if obs_vector is not None and data['ppo'] is not None:
                            action_vivo, _ = data['ppo'].predict(obs_vector, deterministic=True)
                            action_vivo = int(action_vivo)
                            
                            tipo_pos_actual = posiciones[0].type 

                            # Evaluación de Acción 3 con validación de min_velas
                            min_velas_req = 5 if s in ['USDCAD', 'USDCHF', 'AUDUSD'] else 4
                            es_giro_opuesto = (tipo_pos_actual == 0 and action_vivo == 2) or (tipo_pos_actual == 1 and action_vivo == 1)
                            es_accion_tres_valida = (action_vivo == 3 and velas_en_posicion >= min_velas_req)

                            if es_giro_opuesto or es_accion_tres_valida:
                                motivo_cierre = "PPO dictamina Giro Opuesto" if es_giro_opuesto else "PPO dictamina Acción 3 (Cierre Estratégico)"
                                ejecutar_cierre_mercado(s, posiciones[0], motivo_cierre, rsi_now, adx_now)
                                status = "🚨 CIERRE PPO"

                                # En caso de giro opuesto, NO se abre la opuesta en la misma iteración (misma lógica que el backtest)
                else:
                    posiciones_activas[s] = None
                    if par_afectado_por_noticia(s):
                        status = "⚠️ PAUSA NOTICIA"
                        registrar_log(f"⚠️ [{s}] OPERATIVA PAUSADA: Evento económico de alto impacto detectado en la divisa.")
                    elif ultima_vela_procesada[s] != timestamp_vela:
                        # Evaluación directa del PPO previa a la toma de decisiones
                        obs_vector = generar_vector_observacion(df.iloc[:-1], pred_class, s)
                        action_ppo = 0
                        if obs_vector is not None and data['ppo'] is not None:
                            action_ppo, _ = data['ppo'].predict(obs_vector, deterministic=True)
                            action_ppo = int(action_ppo)

                        # Exigir estrictamente la convergencia LSTM + PPO (pred_class == action_ppo)
                        if pred_class == 1 and action_ppo == 1:
                            if REGLAS_PAR.get(s, {}).get('use_macro_filter', False) and close_actual < ema_200_actual:
                                registrar_log(f"⚠️ [{s}] VETO MACRO: Sugerencia COMPRA pero el precio ({close_actual:.5f}) está por debajo de la EMA 200 ({ema_200_actual:.5f}).")
                                status = "❌ VETO EMA200 (Macro bajista)"
                            else:
                                registrar_log(f"[{s}] ADVERTENCIA COMPRA -> ADX: {adx_now:.1f}, RSI: {rsi_now:.1f}, CLASE LSTM: {pred_class}, PPO: {action_ppo}")
                                res_orden = enviar_orden(s, mt5.ORDER_TYPE_BUY, pred_class, data['ppo'], df.iloc[:-1], action_ppo)
                                if res_orden and res_orden.retcode == mt5.TRADE_RETCODE_DONE:
                                    status = "✅ COMPRA"
                                else:
                                    status = "❌ VETADO PPO/FAIL"
                            
                        elif pred_class == 2 and action_ppo == 2:
                            if REGLAS_PAR.get(s, {}).get('use_macro_filter', False) and close_actual > ema_200_actual:
                                registrar_log(f"⚠️ [{s}] VETO MACRO: Sugerencia VENTA pero el precio ({close_actual:.5f}) está por encima de la EMA 200 ({ema_200_actual:.5f}).")
                                status = "❌ VETO EMA200 (Macro alcista)"
                            else:
                                registrar_log(f"[{s}] ADVERTENCIA VENTA -> ADX: {adx_now:.1f}, RSI: {rsi_now:.1f}, CLASE LSTM: {pred_class}, PPO: {action_ppo}")
                                res_orden = enviar_orden(s, mt5.ORDER_TYPE_SELL, pred_class, data['ppo'], df.iloc[:-1], action_ppo)
                                if res_orden and res_orden.retcode == mt5.TRADE_RETCODE_DONE:
                                    status = "✅ VENTA"
                                else:
                                    status = "❌ VETADO PPO/FAIL"
                        else:
                            status = f"⌛ LATERAL ({adx_now:.1f})"
                        
                        # CONGELAR VELA: Marca la vela como procesada para no sobre-operar iterativamente
                        ultima_vela_procesada[s] = timestamp_vela
                    else: 
                        status = f"⌛ ESPERANDO NUEVA VELA"

            estado_radar[s] = {"status": status, "adx": adx_now, "ppo_status": current_ppo_status, "ppo": data['ppo']}
            
            adx_str = f"ADX: {adx_now:.1f}"
            pred_str = f"CLASS: {pred_class}"
            ppo_str = f"[PPO: {estado_radar[s]['ppo_status']}]"
            print(f"{s:<12} | {adx_str:<9} | {pred_str:<11} | {ppo_str:<13} | {status}")
            
        print("─" * 70)
        time.sleep(10)

def run_telegram_bot():
    try:
        app = Application.builder().token(TELEGRAM_TOKEN.strip()).post_init(configurar_menu_comandos).build()
        app.add_handler(CommandHandler("status", cmd_status))
        app.add_handler(CommandHandler("radar", cmd_radar))
        app.add_handler(CommandHandler("pause", cmd_pause))
        app.add_handler(CommandHandler("resume", cmd_resume))
        app.add_handler(CommandHandler("kill_system", cmd_kill_system))
        app.add_handler(CallbackQueryHandler(callback_button_handler))
        app.run_polling()
    except Exception as e:
        pass

if __name__ == "__main__":
    mostrar_lanzador()
    
    if mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        cargar_infraestructura()
        
        t_bot = threading.Thread(target=run_telegram_bot, daemon=True)
        t_bot.start()
        
        bucle_radar()