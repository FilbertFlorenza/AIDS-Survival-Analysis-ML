# Gunakan image Python yang ringan
FROM python:3.11-slim

# Tetapkan direktori kerja di dalam kontainer
WORKDIR /app

# Salin hanya requirements terlebih dahulu untuk optimasi cache Docker
COPY requirements.txt .

# Instal dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Salin sisa kode aplikasi
COPY . .

# Ekspos port default Streamlit
EXPOSE 8501

# Tambahkan Healthcheck untuk monitoring status kontainer
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Perintah eksekusi utama (Penting: bind ke 0.0.0.0)
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]