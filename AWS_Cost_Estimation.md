# Cotización AWS — ChismesitoGPT / Agentes IRIS

## Servicios identificados

| # | Servicio AWS | Rol en la app | Modelo de cobro principal |
|---|-------------|---------------|--------------------------|
| 1 | **Amazon Bedrock** | LLM orquestador (Claude) + SLM (Whisper) | Tokens entrada/salida |
| 2 | **Amazon S3** | Modelos pequeños (spaCy), archivos temporales | GB/mes almacenados + requests |
| 3 | **AWS Lambda** | Llamadas a APIs, scraping ligero, procesos serverless | Requests + GB-segundo |
| 4 | **AWS Amplify** | Frontend de ChismesitoGPT | Build minutes + GB served |
| 5 | **OpenSearch Serverless** | Memoria vectorial de conversaciones (RAG) | OCU (OpenSearch Compute Units)/hora |
| 6 | **AWS RDS PostgreSQL** | Base relacional de comentarios scrapeados | vCPU + RAM + GB almacenados |
| 7 | **AWS Kendra** | Búsqueda web para scraping | Horas de conector + queries + documentos |
| 8 | **Amazon Transcribe** | Speech-to-text (futuro) | Minutos transcritos |
| 9 | **Amazon Rekognition** | Análisis de imágenes/videos de posts | Imágenes procesadas + minutos de video |
| 10 | **ECS Fargate** | Scraping pesado (Chrome headless, APIFY) | vCPU/hora + GB RAM/hora |
| 11 | **Elastic Load Balancing (ALB)** | Balanceo HTTPS/WebSocket | LCU/hora (capacity units) |
| 12 | **Secrets Manager** | Guardar API keys (Gemini, Twitter, etc.) | $/secreto/mes + $/10k llamadas |
| 13 | **Route 53** | DNS | $/zona alojada/mes + $/millón queries |
| 14 | **CloudWatch** | Logs y monitoreo | GB ingeridos + GB almacenados |

---

## Variables del modelo de costos

| Variable | Descripción | Unidad |
|----------|-------------|--------|
| **N** | Número de usuarios activos/mes | usuarios |
| **Q** | Queries/scrapeos promedio por usuario por mes | queries/usuario/mes |
| **C** | Comentarios promedio scrapeados por query | comentarios/query |
| **T_in** | Tokens de entrada promedio por query (prompt + contexto) | tokens |
| **T_out** | Tokens de salida promedio por query (respuesta + categorías) | tokens |
| **L** | Tamaño promedio de logs por query | MB |

---

## Modelo de costos por servicio

### 1. Amazon Bedrock (LLM)

```
Costo_Bedrock = N × Q × [(T_in × Precio_in) + (T_out × Precio_out)]
```

| Modelo | Precio 1K tokens in | Precio 1K tokens out |
|--------|--------------------|--------------------|
| Claude 3.5 Sonnet | $0.003 | $0.015 |
| Claude 3 Haiku (económico) | $0.00025 | $0.00125 |
| Amazon Titan Text | $0.0003 | $0.0004 |

**Ejemplo (Claude Haiku)**: N=100, Q=10, T_in=3000, T_out=500
→ 100 × 10 × [(3 × $0.00025) + (0.5 × $0.00125)] = $1.38/mes

### 2. Amazon S3 (Almacenamiento)

```
Costo_S3 = (Modelos_MB / 1024 × Precio_GB) + (N × Q × Archivos_por_query × Precio_PUT)
```

| Concepto | Precio |
|----------|--------|
| Almacenamiento Standard | $0.023 / GB/mes |
| PUT/COPY/POST requests | $0.005 / 1000 |
| GET requests | $0.0004 / 1000 |

**Ejemplo**: 2 GB modelos + 1000 users × 10 queries × 0.0001 GB × PUT
→ (2 × $0.023) + (1000 × 10 × 0.005/1000) = $0.05 + $0.05 = ~$0.10/mes

### 3. AWS Lambda

```
Costo_Lambda = N × Q × [Requests × $0.20/1M + Duración_s × Memoria_GB × $0.0000166667/GB-s]
```

| Concepto | Precio |
|----------|--------|
| Requests | $0.20 / millón |
| Duración | $0.0000166667 / GB-segundo |

**Ejemplo (256 MB, 3s ejecución, 1 request por query)**:
N=100, Q=10
→ (1000 × $0.20/1M) + (1000 × 3s × 0.25GB × $0.00001667) = $0.0002 + $0.0125 = ~$0.01/mes

### 4. AWS Amplify

```
Costo_Amplify = Build_minutos × $0.01 + GB_servidos × $0.15 + (N × GB_visitante)
```

| Concepto | Precio |
|----------|--------|
| Build & deploy | $0.01 / minuto |
| GB served | $0.15 / GB |
| Free tier | 1000 build min/mes, 5 GB/mes |

**Alternativa más barata: S3 + CloudFront** → $0.085/GB transfer (primeros 10 TB)

### 5. OpenSearch Serverless (Vector DB)

```
Costo_OpenSearch = OCU_horas × $0.24/hora
```

| Concepto | Precio |
|----------|--------|
| Search OCU | $0.24 / hora |
| Indexing OCU | $0.24 / hora |
| Mínimo | 4 OCU (2 search + 2 indexing) = $0.96/hora |

**24/7 operación**: 4 OCU × 730h × $0.24 = **$700.80/mes** (⚠️ caro para MVP)

**Alternativa**: Supabase pgvector (incluido en plan Supabase Pro ~$25/mes) o Amazon RDS con extensión pgvector.

### 6. AWS RDS PostgreSQL

```
Costo_RDS = Instancia_horas + Almacenamiento_GB × $0.115 + (N × Q × C × 0.5KB / 1M × IOPS)
```

| Tamaño | vCPU | RAM | Precio/hr (on-demand) | Precio/mes |
|--------|------|-----|----------------------|-----------|
| db.t3.micro | 2 | 1 GB | $0.016 | ~$12 |
| db.t3.small | 2 | 2 GB | $0.034 | ~$25 |
| db.t3.medium | 2 | 4 GB | $0.068 | ~$50 |
| Aurora Serverless v2 | 0.5-128 ACU | — | $0.12/ACU-hr | ~$44 min |

**Ejemplo (db.t3.small + 20 GB)**: $25 + (20 × $0.115) = **$27.30/mes**

### 7. AWS Kendra

```
Costo_Kendra = Conectores_horas × $1.00 + (N × Q × 0.001_scan) + Docs_indexados × $0.0004/hora
```

| Concepto | Precio |
|----------|--------|
| Connector uso | $1.00 / hora |
| Queries escaneadas | $0.001 / query |
| Documentos indexados (primeros 100k) | $0.0004 / hora |

⚠️ AWS Kendra tiene un costo base **alto**: ~$1,008/mes (índice Enterprise mínimo para web crawling).

**Alternativa**: SerpAPI ($50/mes plan básico) o Google Custom Search (gratis 100 queries/día).

### 8. Amazon Transcribe (futuro)

```
Costo_Transcribe = N × Minutos_transcritos × Precio_minuto
```

| Concepto | Precio |
|----------|--------|
| Standard | $0.024 / minuto |
| Free tier | 60 min/mes por 12 meses |

### 9. Amazon Rekognition

```
Costo_Rekognition = N × Q × Imágenes_por_query × Precio_imagen
```

| Concepto | Precio |
|----------|--------|
| Image analysis | $0.001 / imagen (primeros 1M) |
| Video analysis | $0.10 / minuto |
| Free tier | 5000 imágenes/mes |

**Ejemplo**: N=100, Q=10, 5 imágenes/query = 5000 imágenes → **$5.00/mes** (o $0 con free tier)

### 10. ECS Fargate (scraping pesado)

```
Costo_Fargate = Tareas × (vCPU_horas × $0.04048 + GB_RAM_horas × $0.004445)
```

| Configuración | vCPU | RAM | Precio/hr | Precio/mes (8h/día) |
|--------------|------|-----|----------|---------------------|
| Pequeño | 1 | 2 GB | $0.049 | ~$11.80 |
| Mediano | 2 | 4 GB | $0.099 | ~$23.80 |
| Grande | 4 | 8 GB | $0.198 | ~$47.50 |

⚠️ Solo necesario para scraping pesado con Chrome/Selenium. Con APIFY y APIs oficiales, se puede reemplazar por Lambda.

### 11. Application Load Balancer (ALB)

```
Costo_ALB = ALB_horas × $0.0225 + LCU_horas × $0.008
```

| Concepto | Precio |
|----------|--------|
| ALB por hora | $0.0225 |
| LCU (capacity units) | $0.008 / hora |
| Mínimo | ~1 LCU = $0.0305/hora |

**Costo mínimo 24/7**: **~$22/mes**

### 12. Secrets Manager

```
Costo_Secrets = Secretos × $0.40 + (N × $0.05/10K_llamadas)
```

| Concepto | Precio |
|----------|--------|
| Por secreto/mes | $0.40 |
| Por 10,000 llamadas | $0.05 |

**Ejemplo (15 secretos)**: 15 × $0.40 = **$6.00/mes**

**Alternativa más barata**: Parameter Store (Standard tier es gratis, Advanced $0.05/parámetro/mes).

### 13. Route 53

| Concepto | Precio |
|----------|--------|
| Zona alojada | $0.50 / mes |
| Queries (primeros 1B) | $0.40 / millón |

**Costo fijo**: **~$0.50/mes** por zona

### 14. CloudWatch

| Concepto | Precio |
|----------|--------|
| Log ingestion | $0.50 / GB |
| Log storage | $0.03 / GB/mes |
| Metrics | $0.30 / métrica/mes |

```
Costo_CloudWatch = (N × Q × L_MB / 1024 × $0.50) + Logs_acumulados_GB × $0.03
```

---

## Fórmula de Costo Total

```
Costo_Total(N, Q) = 
    Costo_Bedrock(N, Q, T_in, T_out)
  + Costo_S3(N, Q)
  + Costo_Lambda(N, Q)
  + Costo_Amplify(N)
  + Costo_OpenSearch  (o Costo_Supabase_pgvector)
  + Costo_RDS
  + Costo_Kendra      (o Costo_SerpAPI)
  + Costo_Rekognition(N, Q)
  + Costo_ECS_Fargate(N, Q)  (solo si no se usa APIFY/Lambda)
  + Costo_ALB
  + Costo_Secrets
  + Costo_Route53
  + Costo_CloudWatch(N, Q)
```

---

## Tabla de cotización con escenarios

### Supuestos base:
- Q = 10 queries/usuario/mes (1 búsqueda cada 3 días)
- C = 50 comentarios promedio por query
- T_in = 3,000 tokens, T_out = 500 tokens por query
- Modelo LLM: Claude 3 Haiku (económico)
- RDS: db.t3.small (2 vCPU, 2 GB)
- OpenSearch: reemplazado por Supabase pgvector ($25/mes Pro)
- Kendra: reemplazado por SerpAPI ($50/mes)
- ECS: solo Lambda (sin scraping via Chrome)

| Servicio | N=10 | N=50 | N=100 | N=500 | N=1,000 |
|----------|------|------|-------|-------|---------|
| **Bedrock (Claude Haiku)** | $0.14 | $0.69 | $1.38 | $6.88 | $13.75 |
| **S3** | $0.06 | $0.06 | $0.10 | $0.25 | $0.50 |
| **Lambda** | $0.00 | $0.01 | $0.01 | $0.06 | $0.13 |
| **Amplify** | $0.00* | $0.00* | $0.00* | $0.00* | $0.00* |
| **Supabase pgvector** | $25.00 | $25.00 | $25.00 | $25.00 | $25.00 |
| **RDS PostgreSQL** | $27.30 | $27.30 | $27.30 | $27.30 | $27.30 |
| **SerpAPI** | $50.00 | $50.00 | $50.00 | $50.00 | $50.00 |
| **Rekognition** | $0.00 | $0.25 | $0.50 | $2.50 | $5.00 |
| **ECS Fargate** | $0.00 | $0.00 | $0.00 | $0.00 | $0.00 |
| **ALB** | $22.00 | $22.00 | $22.00 | $22.00 | $22.00 |
| **Secrets Manager** | $6.00 | $6.00 | $6.00 | $6.00 | $6.00 |
| **Route 53** | $0.50 | $0.50 | $0.50 | $0.50 | $0.50 |
| **CloudWatch** | $0.10 | $0.50 | $1.00 | $5.00 | $10.00 |
| | | | | | |
| **TOTAL/mes** | **$131.10** | **$132.31** | **$133.79** | **$145.49** | **$160.18** |
| **Costo por usuario** | $13.11 | $2.65 | $1.34 | $0.29 | $0.16 |

*\* Amplify free tier: 1000 build min + 5 GB/mes*

---

## Gráfico: Costo Total vs Usuarios

```
Costo/mes
  $160 |                                    ●
  $155 |
  $150 |                         ●
  $145 |
  $140 |
  $135 |              ●
  $130 |    ●     ●
       +-----+-----+-----+-----+-----+---> Usuarios
        10    50   100   500   1000
```

**Interpretación**: El costo está dominado por los servicios fijos (~$130 base). La parte variable (usuarios) crece muy lentamente.

---

## Recomendaciones para reducir costos

| Servicio caro | Alternativa | Ahorro estimado |
|--------------|-------------|-----------------|
| OpenSearch ($700/mes) | Supabase pgvector (~$25/mes) | **-$675/mes** |
| Kendra ($1,008/mes) | SerpAPI ($50/mes) + Google CS (gratis) | **-$958/mes** |
| Secrets Manager ($6) | AWS Parameter Store (gratis) | **-$6/mes** |
| RDS stand-alone | Aurora Serverless v2 (escala a 0) | **-$10/mes** |
| Bedrock Claude Sonnet | Claude Haiku (10× más barato) | **-90% tokens** |

### Con todas las optimizaciones aplicadas:

| N usuarios | Costo base optimizado |
|-----------|----------------------|
| 10 | **~$120/mes** |
| 50 | **~$121/mes** |
| 100 | **~$123/mes** |
| 500 | **~$135/mes** |
| 1000 | **~$150/mes** |

---

## Proceso de estimación para cotización

### Paso 1: Definir parámetros del cliente
```
□ N = Número de usuarios activos esperados
□ Q = Frecuencia de búsquedas por usuario al mes
□ Plataformas objetivo (YouTube, Twitter, FB, etc.)
□ ¿Necesitan transcripción de audio? (Sí/No)
□ ¿Necesitan análisis de imágenes? (Sí/No)
□ ¿Volumen de scraping pesado? (Alto/Medio/Bajo)
```

### Paso 2: Seleccionar tier de servicios
```
□ LLM: [ ] Haiku (económico)  [ ] Sonnet (balance)  [ ] Opus (premium)
□ DB:  [ ] Aurora Serverless  [ ] RDS t3.small     [ ] RDS t3.large
□ Search: [ ] SerpAPI $50/mes  [ ] Kendra Enterprise
□ Vector: [ ] Supabase pgvector  [ ] OpenSearch Serverless
```

### Paso 3: Aplicar fórmula `Costo_Total(N, Q)`

### Paso 4: Agregar margen (20-30%) para imprevistos y soporte

### Paso 5: Generar PDF de cotización con tabla de costos mensuales + anuales

---

## Plantilla de cotización rápida

```
=================================================================
COTIZACIÓN AWS — ChismesitoGPT / Agentes IRIS
=================================================================
Cliente: ________________________
Fecha:   ________________________
Usuarios estimados: N = ______
Queries/usuario/mes: Q = ______

SERVICIOS AWS (estimado mensual):
-----------------------------------------------------
Amazon Bedrock (LLM Claude) ........ $ ________
Amazon S3 (almacenamiento) ......... $ ________
AWS Lambda (serverless) ........... $ ________
AWS Amplify (frontend) ............ $ ________
Supabase pgvector (vector DB) ..... $ ________
AWS RDS PostgreSQL ................ $ ________
SerpAPI (búsqueda web) ............ $ ________
Amazon Rekognition (imágenes) ..... $ ________
ECS Fargate (scraping) ............ $ ________
Application Load Balancer ......... $ ________
AWS Secrets Manager ............... $ ________
Route 53 (DNS) .................... $ ________
CloudWatch (monitoreo) ............ $ ________
-----------------------------------------------------
SUBTOTAL AWS ...................... $ ________
Margen operativo (25%) ............ $ ________
-----------------------------------------------------
TOTAL MENSUAL ..................... $ ________
TOTAL ANUAL ....................... $ ________
=================================================================
```
