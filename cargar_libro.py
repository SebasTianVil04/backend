"""Carga senas.csv -> categorias, lecciones, clases (solo video + practica por vision).

Uso (desde la carpeta con los dos archivos):
  docker compose cp cargar_libro.py backend:/app/cargar_libro.py
  docker compose cp senas.csv backend:/app/senas.csv
  docker compose exec backend python /app/cargar_libro.py

Conecta con DIRECT_URL (signafree_owner); si no existe, usa DATABASE_URL.
Idempotente: se puede correr varias veces y no borra videos ya cargados.
"""
import csv
import os
import re
import sys
from pathlib import Path

try:
    import psycopg2 as pg
except ImportError:
    import psycopg as pg  # psycopg 3

BASE = Path(__file__).resolve().parent
CSV_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "senas.csv"

url = (os.environ.get("DIRECT_URL") or os.environ.get("direct_url")
       or os.environ.get("DATABASE_URL") or os.environ.get("database_url"))
if not url:
    raise SystemExit("Falta DIRECT_URL (o DATABASE_URL) en el entorno del contenedor")
url = re.sub(r"^postgres(?:ql)?(?:\+\w+)?://", "postgresql://", url)

if not CSV_PATH.exists():
    raise SystemExit(f"No existe {CSV_PATH}. Copia senas.csv con: docker compose cp senas.csv backend:/app/senas.csv")

SQL_STG = """
CREATE FUNCTION pg_temp.norm(t text) RETURNS text LANGUAGE sql IMMUTABLE AS $$
  SELECT trim(both '_' from regexp_replace(translate(upper(t), 'ÁÉÍÓÚÜ', 'AEIOUU'), '[^A-Z0-9Ñ]+', '_', 'g'))
$$;
CREATE TEMP TABLE stg (rn serial, capitulo int, leccion text, sena text, video_url text, practica text);
"""

SQL_MAIN = r"""
CREATE TEMP TABLE caps (cap int, nombre text, tipo text, icono text, color text);
INSERT INTO caps VALUES
 (1,'Alfabeto','alfabeto','bi-alphabet','#3B82F6'),
 (2,'Números','numeros','bi-123','#F59E0B'),
 (3,'Familia','vocabulario','bi-people','#8B5CF6'),
 (4,'Cuerpo Humano y Primeros Auxilios','vocabulario','bi-heart-pulse','#EF4444'),
 (5,'Prendas de Vestir y Accesorios','vocabulario','bi-bag','#EC4899'),
 (6,'Alimentos','vocabulario','bi-egg-fried','#F97316'),
 (7,'Frutas','vocabulario','bi-apple','#22C55E'),
 (8,'Verduras','vocabulario','bi-flower1','#16A34A'),
 (9,'Bebidas','vocabulario','bi-cup-straw','#0EA5E9'),
 (10,'Adverbios y Preposiciones','vocabulario','bi-arrows-move','#6366F1'),
 (11,'Preguntas','vocabulario','bi-question-circle','#6366F1'),
 (12,'Pronombres y Signos Gramaticales','vocabulario','bi-person-lines-fill','#6366F1'),
 (13,'Verbos','vocabulario','bi-lightning','#6366F1'),
 (14,'Adjetivos','vocabulario','bi-card-text','#6366F1'),
 (15,'Animales','vocabulario','bi-bug','#84CC16'),
 (16,'Juguetes, Juegos y Deportes','vocabulario','bi-dribbble','#F59E0B'),
 (17,'Profesiones y Oficios','vocabulario','bi-briefcase','#64748B'),
 (18,'Países y Lugares','vocabulario','bi-globe-americas','#0EA5E9'),
 (19,'Religión','vocabulario','bi-brightness-high','#A855F7'),
 (20,'Ambientes y Artículos del Hogar','vocabulario','bi-house','#78716C'),
 (21,'Nociones Temporales','vocabulario','bi-clock','#14B8A6'),
 (22,'Naturaleza','vocabulario','bi-tree','#22C55E'),
 (23,'Colegio y Colores','vocabulario','bi-pencil','#EC4899'),
 (24,'Informática','vocabulario','bi-laptop','#3B82F6');

INSERT INTO tipos_categoria (valor, etiqueta, icono, color, activo) VALUES
 ('alfabeto','Alfabeto','bi-alphabet','#3B82F6',true),
 ('numeros','Números','bi-123','#F59E0B',true),
 ('vocabulario','Vocabulario','bi-book','#10B981',true)
ON CONFLICT (valor) DO NOTHING;

INSERT INTO categorias (nombre, tipo_id, descripcion, icono, color, orden, nivel_requerido, activa)
SELECT c.nombre, t.id, c.nombre, c.icono, c.color, 10 + c.cap, 1, true
FROM caps c
JOIN tipos_categoria t ON t.valor = c.tipo
WHERE c.cap IN (SELECT DISTINCT capitulo FROM stg)
ON CONFLICT (nombre) DO NOTHING;

CREATE TEMP TABLE s AS
SELECT rn, capitulo, trim(sena) AS sena,
       nullif(trim(video_url), '') AS video_url,
       nullif(trim(practica), '')  AS practica,
       CASE WHEN nullif(trim(leccion), '') IS NOT NULL THEN trim(leccion)
            ELSE (SELECT nombre FROM caps WHERE cap = capitulo) || ' ' ||
                 ((row_number() OVER (PARTITION BY capitulo, (nullif(trim(leccion), '') IS NULL) ORDER BY rn) - 1) / 8 + 1)
       END AS lec
FROM stg
WHERE nullif(trim(sena), '') IS NOT NULL;

CREATE TEMP TABLE s2 AS
SELECT DISTINCT ON (capitulo, lec, upper(sena)) *
FROM s
ORDER BY capitulo, lec, upper(sena), rn;

CREATE TEMP TABLE l AS
SELECT capitulo, lec, min(rn) AS mrn, count(*) AS n,
       (array_agg(sena ORDER BY rn))[1] || ' - ' || (array_agg(sena ORDER BY rn DESC))[1] AS rango,
       dense_rank() OVER (PARTITION BY capitulo ORDER BY min(rn)) AS orden
FROM s2
GROUP BY capitulo, lec;

INSERT INTO lecciones (categoria_id, titulo, descripcion, sena, orden, nivel_dificultad, activa, bloqueada, puntos_base, puntos_perfecto)
SELECT ca.id, l.lec, l.n || ' señas en video', left(l.rango, 100), l.orden, 1, true, false, 10, 20
FROM l
JOIN caps c ON c.cap = l.capitulo
JOIN categorias ca ON ca.nombre = c.nombre
WHERE NOT EXISTS (SELECT 1 FROM lecciones x WHERE x.categoria_id = ca.id AND x.titulo = l.lec);

UPDATE lecciones x
SET orden = l.orden, sena = left(l.rango, 100)
FROM l
JOIN caps c ON c.cap = l.capitulo
JOIN categorias ca ON ca.nombre = c.nombre
WHERE x.categoria_id = ca.id AND x.titulo = l.lec;

UPDATE lecciones x
SET leccion_previa_id = p.id, bloqueada = true
FROM l
JOIN caps c ON c.cap = l.capitulo
JOIN categorias ca ON ca.nombre = c.nombre
JOIN l lp ON lp.capitulo = l.capitulo AND lp.orden = l.orden - 1
JOIN lecciones p ON p.categoria_id = ca.id AND p.titulo = lp.lec
WHERE x.categoria_id = ca.id AND x.titulo = l.lec;

CREATE TEMP TABLE modelo AS
SELECT DISTINCT pg_temp.norm(v) AS cod FROM (
  SELECT jsonb_array_elements_text(clases_json::jsonb) AS v
  FROM modelos_ia
  WHERE activo AND clases_json IS NOT NULL AND jsonb_typeof(clases_json::jsonb) = 'array'
  UNION ALL
  SELECT e.value FROM modelos_ia m, jsonb_each_text(m.clases_json::jsonb) e
  WHERE m.activo AND m.clases_json IS NOT NULL AND jsonb_typeof(m.clases_json::jsonb) = 'object'
  UNION ALL
  SELECT e.key FROM modelos_ia m, jsonb_each_text(m.clases_json::jsonb) e
  WHERE m.activo AND m.clases_json IS NOT NULL AND jsonb_typeof(m.clases_json::jsonb) = 'object'
) q;

CREATE TEMP TABLE k AS
SELECT le.id AS leccion_id, s.sena, left(pg_temp.norm(s.sena), 50) AS cod, s.video_url,
       row_number() OVER (PARTITION BY s.capitulo, s.lec ORDER BY s.rn) AS orden,
       CASE WHEN s.video_url ~* 'youtu' THEN 'YOUTUBE'
            WHEN s.video_url ~* 'drive\.google' THEN 'GOOGLE_DRIVE'
            WHEN s.video_url ~* 'vimeo' THEN 'VIMEO' END AS tipo,
       CASE WHEN s.video_url ~* 'youtu' THEN substring(s.video_url from '(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})')
            WHEN s.video_url ~* 'drive\.google' THEN coalesce(substring(s.video_url from '/d/([A-Za-z0-9_-]+)'),
                                                              substring(s.video_url from '[?&]id=([A-Za-z0-9_-]+)'))
            WHEN s.video_url ~* 'vimeo' THEN substring(s.video_url from 'vimeo\.com/(?:video/)?([0-9]+)') END AS vid,
       CASE WHEN s.practica IS NOT NULL THEN s.practica = '1'
            ELSE left(pg_temp.norm(s.sena), 50) IN (SELECT cod FROM modelo) END AS prac
FROM s2 s
JOIN caps c ON c.cap = s.capitulo
JOIN categorias ca ON ca.nombre = c.nombre
JOIN lecciones le ON le.categoria_id = ca.id AND le.titulo = s.lec;

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM k WHERE video_url IS NOT NULL AND (tipo IS NULL OR vid IS NULL)) THEN
    RAISE EXCEPTION 'URL de video no reconocida: %',
      (SELECT string_agg(sena || ' -> ' || video_url, '; ') FROM k WHERE video_url IS NOT NULL AND (tipo IS NULL OR vid IS NULL));
  END IF;
END
$$;

INSERT INTO clases (leccion_id, titulo, descripcion, sena, tipo_video, video_url, video_id, orden,
                    duracion_estimada, requiere_practica, intentos_minimos, precision_minima, activa)
SELECT leccion_id, left(sena, 255), 'Seña: ' || left(sena, 200), cod, coalesce(tipo, 'YOUTUBE')::tipovideo, video_url, vid, orden,
       60, prac, 3, 0.70, true
FROM k
WHERE NOT EXISTS (SELECT 1 FROM clases x WHERE x.leccion_id = k.leccion_id AND x.titulo = left(k.sena, 255));

UPDATE clases x
SET orden = k.orden, sena = k.cod,
    requiere_practica = k.prac,
    intentos_minimos = 3,
    precision_minima = 0.70,
    tipo_video = coalesce(k.tipo::tipovideo, x.tipo_video, 'YOUTUBE'::tipovideo),
    video_url  = coalesce(k.video_url, x.video_url),
    video_id   = coalesce(k.vid, x.video_id)
FROM k
WHERE x.leccion_id = k.leccion_id AND x.titulo = left(k.sena, 255);
"""

SQL_RESUMEN = """
SELECT c.nombre, count(DISTINCT le.id), count(cl.id), count(cl.video_url),
       count(*) FILTER (WHERE cl.requiere_practica)
FROM categorias c
JOIN lecciones le ON le.categoria_id = c.id
JOIN clases cl ON cl.leccion_id = le.id
WHERE c.orden BETWEEN 11 AND 34
GROUP BY c.nombre, c.orden
ORDER BY c.orden
"""

filas = []
with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        sena = (r.get("sena") or "").strip()
        if not sena:
            continue
        filas.append((int(r["capitulo"]), (r.get("leccion") or "").strip(), sena,
                      (r.get("video_url") or "").strip(), (r.get("practica") or "").strip()))
print(f"CSV: {len(filas)} señas")

conn = pg.connect(url)
try:
    cur = conn.cursor()
    cur.execute(SQL_STG)
    cur.executemany(
        "INSERT INTO stg (capitulo, leccion, sena, video_url, practica) VALUES (%s,%s,%s,%s,%s)", filas)
    cur.execute(SQL_MAIN)
    conn.commit()
    cur.execute(SQL_RESUMEN)
    print(f"{'Categoría':<36}{'Lecc.':>6}{'Clases':>8}{'Video':>7}{'Práct.':>8}")
    tot = [0, 0, 0, 0]
    for nombre, nl, nc, nv, npr in cur.fetchall():
        print(f"{nombre:<36}{nl:>6}{nc:>8}{nv:>7}{npr:>8}")
        for i, v in enumerate((nl, nc, nv, npr)):
            tot[i] += v
    print(f"{'TOTAL':<36}{tot[0]:>6}{tot[1]:>8}{tot[2]:>7}{tot[3]:>8}")
except Exception:
    conn.rollback()
    raise
finally:
    conn.close()
