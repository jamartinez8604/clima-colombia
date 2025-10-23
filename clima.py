# ==========================================#
#  ANALISIS DE DATOS CLIMATICOS (clima.csv) #
# ==========================================#

from pyspark.sql import SparkSession, functions as F
import unicodedata, re, os, shutil

spark = (SparkSession.builder.appName("Analisis_Clima_Local").getOrCreate())
spark.sparkContext.setLogLevel("WARN")

# ========== 1) Cargar desde HDFS (solo lectura)
file_path_hdfs = "hdfs://localhost:9000/Tarea3_Procesamiento_Datos/clima.csv"
df = (spark.read
      .format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(file_path_hdfs))

print("\n=== ESQUEMA ORIGINAL ===")
df.printSchema()
df.show(5, truncate=False)

# ========== 2) Normalizar nombres de columnas
def normalize_col(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = s.replace(" ", "_")
    s = s.replace("__", "_")
    s = re.sub(r"[^a-z0-9_]", "_", s)  # solo a-z0-9_
    s = re.sub(r"_+", "_", s).strip("_")
    return s

for c in df.columns:
    df = df.withColumnRenamed(c, normalize_col(c))

print("\n=== ESQUEMA NORMALIZADO ===")
df.printSchema()
df.show(5, truncate=False)

# Esperados tras normalizar (ejemplos):
# periodo, parametro, codigo, categoria, estacion, municipio, departamento, ao,
# altitud_m, longitud, latitud, ene, feb, mar, abr, may, jun, jul, ago, sep, oct, nov, dic, anual

# ========== 3) Tipificar columnas numéricas (meses y coords)
meses = ["ene","feb","mar","abr","may","jun","jul","ago","sep","oct","nov","dic"]
numericas = ["longitud","latitud"] + meses + ["anual"]

for col in numericas:
    if col in df.columns:
        df = df.withColumn(col, F.regexp_replace(F.col(col).cast("string"), ",", ""))
        df = df.withColumn(col, F.col(col).cast("double"))

if "altitud_m" in df.columns:
    df = df.withColumn("altitud_m", F.regexp_replace(F.col("altitud_m").cast("string"), ",", ""))
    df = df.withColumn("altitud_m", F.col("altitud_m").cast("double"))

# ========== 4) Limpieza básica
subset_req = [c for c in ["departamento"] if c in df.columns]
df = df.dropDuplicates()
if subset_req:
    df = df.dropna(subset=subset_req)

# ========== 5) SOLO: Promedio mensual por departamento
if not all(m in df.columns for m in meses):
    raise ValueError("El archivo no contiene las 12 columnas de meses (ene..dic).")

# Unpivot ancho -> largo (mes, valor) y calcular promedio
pairs = []
for m in meses:
    pairs.extend([F.lit(m), F.col(m)])

df_long = (
    df.select("departamento", *meses)
      .select("departamento", F.explode(F.create_map(*pairs)).alias("mes","valor"))
)

promedio_mensual = (df_long.groupBy("departamento","mes")
                    .agg(F.avg("valor").alias("promedio_mensual"))
                    .orderBy("departamento","mes"))

# ========== 6) Guardar en DISCO LOCAL y crear ZIP
OUT_BASE_LOCAL = "/home/vboxuser/Downloads/resultado_clima_normales"

# limpiar y recrear carpeta
if os.path.isdir(OUT_BASE_LOCAL):
    shutil.rmtree(OUT_BASE_LOCAL)
os.makedirs(OUT_BASE_LOCAL, exist_ok=True)

# Guardar SOLO el promedio mensual por departamento
(promedio_mensual.coalesce(1)
    .write.mode("overwrite")
    .option("header","true")
    .csv(f"file://{OUT_BASE_LOCAL}/mensual_por_departamento"))

# Comprimir a ZIP
ZIP_PATH = OUT_BASE_LOCAL + ".zip"
if os.path.exists(ZIP_PATH):
    os.remove(ZIP_PATH)
shutil.make_archive(OUT_BASE_LOCAL, "zip", OUT_BASE_LOCAL)

print(f"\n✅ Resultados escritos en carpeta local: {OUT_BASE_LOCAL}")
print(f"✅ ZIP listo para descargar: {ZIP_PATH}")

spark.stop()
