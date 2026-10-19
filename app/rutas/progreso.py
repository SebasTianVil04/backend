import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, case, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..esquemas.progreso_schemas import (
    ProgresoClaseRespuesta,
    ProgresoLeccionRespuesta,
    RegistroIntentoRequest,
    ResumenDesempenoClase,
)
from ..esquemas.respuestas import RespuestaAPI, RespuestaLista
from ..modelos.clase import Clase
from ..modelos.examen import Examen, ResultadoExamen
from ..modelos.leccion import Leccion
from ..modelos.progreso import ProgresoClase, ProgresoLeccion
from ..modelos.rol import Rol
from ..modelos.usuario import Usuario
from ..utilidades.base_datos import obtener_bd
from ..utilidades.configuracion import configuracion
from ..utilidades.seguridad import obtener_usuario_actual

router = APIRouter(prefix="/progreso", tags=["Progreso"])

XP_POR_NIVEL = 100
PRECISION_MINIMA_DEFECTO = 0.7
ROL_RANKING = "usuario"


def _ahora() -> datetime:
    return datetime.now(timezone.utc)


def _confirmar(db: Session) -> None:
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise


def _calcular_nivel(xp: int) -> int:
    return max(1, (xp // XP_POR_NIVEL) + 1)


def _estrellas(precision: float) -> int:
    if precision >= 0.95:
        return 3
    if precision >= 0.85:
        return 2
    return 1 if precision > 0 else 0


def _normalizar_sena(valor) -> str:
    if valor is None:
        return ""
    texto = unicodedata.normalize("NFD", str(valor).strip().lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return " ".join(texto.split())


def _a_fraccion(valor, defecto: float) -> float:
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return defecto
    return numero / 100.0 if numero > 1 else numero


def _obtener_o_crear_progreso_clase(
    db: Session,
    usuario_id: int,
    clase_id: int,
    vista_inicial: bool = False,
) -> tuple[ProgresoClase, bool]:
    def buscar() -> Optional[ProgresoClase]:
        return (
            db.query(ProgresoClase)
            .filter(ProgresoClase.usuario_id == usuario_id, ProgresoClase.clase_id == clase_id)
            .first()
        )

    progreso = buscar()
    if progreso:
        return progreso, False

    if not db.query(Clase.id).filter(Clase.id == clase_id).first():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Clase no encontrada")

    progreso = ProgresoClase(
        usuario_id=usuario_id,
        clase_id=clase_id,
        vista=vista_inicial,
        completada=False,
        aprobada=False,
        intentos_realizados=0,
        intentos_exitosos=0,
        mejor_precision=0.0,
        puntos_ganados=0,
        xp_ganado=0,
        tiempo_total_practica=0,
    )
    db.add(progreso)

    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        existente = buscar()
        if existente is None:
            raise
        return existente, False

    return progreso, True


def _calcular_puntos_usuario(db: Session, usuario_id: int) -> dict:
    puntos_clases, xp_total = (
        db.query(
            func.coalesce(func.sum(ProgresoClase.puntos_ganados), 0),
            func.coalesce(func.sum(ProgresoClase.xp_ganado), 0),
        )
        .filter(ProgresoClase.usuario_id == usuario_id)
        .one()
    )
    puntos_examenes = (
        db.query(func.coalesce(func.sum(ResultadoExamen.puntuacion_obtenida), 0))
        .filter(ResultadoExamen.usuario_id == usuario_id, ResultadoExamen.aprobado.is_(True))
        .scalar()
    )
    return {
        "puntos_clases": int(puntos_clases),
        "puntos_examenes": int(puntos_examenes),
        "puntos_totales": int(puntos_clases + puntos_examenes),
        "xp_total": int(xp_total),
    }


def _conteos_clases(db: Session, usuario_id: int, leccion_ids: list[int]) -> dict[int, dict]:
    conteos = {i: {"total_clases": 0, "clases_completadas": 0} for i in leccion_ids}
    if not leccion_ids:
        return conteos

    for leccion_id, total in (
        db.query(Clase.leccion_id, func.count(Clase.id))
        .filter(Clase.leccion_id.in_(leccion_ids), Clase.activa.is_(True))
        .group_by(Clase.leccion_id)
        .all()
    ):
        conteos[leccion_id]["total_clases"] = int(total)

    for leccion_id, completadas in (
        db.query(Clase.leccion_id, func.count(ProgresoClase.id))
        .select_from(ProgresoClase)
        .join(Clase, ProgresoClase.clase_id == Clase.id)
        .filter(
            ProgresoClase.usuario_id == usuario_id,
            ProgresoClase.completada.is_(True),
            Clase.activa.is_(True),
            Clase.leccion_id.in_(leccion_ids),
        )
        .group_by(Clase.leccion_id)
        .all()
    ):
        conteos[leccion_id]["clases_completadas"] = int(completadas)

    return conteos


def _metricas_lecciones(db: Session, usuario_id: int, leccion_ids: list[int]) -> dict[int, dict]:
    metricas = _conteos_clases(db, usuario_id, leccion_ids)
    for valores in metricas.values():
        valores.update(
            puntos_clases=0, xp_clases=0, total_intentos=0,
            precision_promedio=0.0, mejor_precision=0.0, puntos_examenes=0,
        )

    if not leccion_ids:
        return metricas

    for leccion_id, puntos, xp, intentos, promedio, mejor in (
        db.query(
            Clase.leccion_id,
            func.coalesce(func.sum(ProgresoClase.puntos_ganados), 0),
            func.coalesce(func.sum(ProgresoClase.xp_ganado), 0),
            func.coalesce(func.sum(ProgresoClase.intentos_realizados), 0),
            func.avg(ProgresoClase.mejor_precision),
            func.max(ProgresoClase.mejor_precision),
        )
        .select_from(ProgresoClase)
        .join(Clase, ProgresoClase.clase_id == Clase.id)
        .filter(
            ProgresoClase.usuario_id == usuario_id,
            Clase.activa.is_(True),
            Clase.leccion_id.in_(leccion_ids),
        )
        .group_by(Clase.leccion_id)
        .all()
    ):
        metricas[leccion_id].update(
            puntos_clases=int(puntos), xp_clases=int(xp), total_intentos=int(intentos),
            precision_promedio=float(promedio or 0), mejor_precision=float(mejor or 0),
        )

    for leccion_id, puntos in (
        db.query(Examen.leccion_id, func.coalesce(func.sum(ResultadoExamen.puntuacion_obtenida), 0))
        .select_from(ResultadoExamen)
        .join(Examen, ResultadoExamen.examen_id == Examen.id)
        .filter(
            ResultadoExamen.usuario_id == usuario_id,
            ResultadoExamen.aprobado.is_(True),
            Examen.leccion_id.in_(leccion_ids),
        )
        .group_by(Examen.leccion_id)
        .all()
    ):
        metricas[leccion_id]["puntos_examenes"] = int(puntos)

    return metricas


def _aplicar_conteos(progreso: ProgresoLeccion, total_clases: int, completadas: int) -> bool:
    ahora = _ahora()
    completada = total_clases > 0 and completadas >= total_clases

    progreso.total_clases = total_clases
    progreso.clases_completadas = completadas
    progreso.porcentaje_completado = round(completadas / total_clases * 100, 1) if total_clases else 0.0

    if completadas > 0 and not progreso.iniciada:
        progreso.iniciada = True
        progreso.fecha_inicio = ahora

    if completada and not progreso.completada:
        progreso.completada = True
        progreso.fecha_completada = ahora
    elif not completada and progreso.completada:
        progreso.completada = False
        progreso.fecha_completada = None

    return completada


@router.get("/clase/{clase_id}", response_model=ProgresoClaseRespuesta)
def obtener_progreso_clase(
    clase_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    progreso, creado = _obtener_o_crear_progreso_clase(db, usuario_actual.id, clase_id)
    if creado:
        _confirmar(db)
        db.refresh(progreso)
    return progreso


@router.post("/clase/{clase_id}/marcar-vista")
def marcar_clase_vista(
    clase_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    progreso, _ = _obtener_o_crear_progreso_clase(db, usuario_actual.id, clase_id)

    if not progreso.vista:
        progreso.vista = True
        progreso.fecha_primera_vista = _ahora()

    _confirmar(db)
    db.refresh(progreso)

    return {
        "mensaje": "Clase marcada como vista",
        "progreso": ProgresoClaseRespuesta.model_validate(progreso),
    }


@router.post("/clases/{clase_id}/practica", response_model=RespuestaAPI)
def guardar_resultado_practica(
    clase_id: int,
    datos: RegistroIntentoRequest,
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_bd),
):
    clase = db.query(Clase).filter(Clase.id == clase_id).first()
    if not clase:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Clase no encontrada")

    progreso, _ = _obtener_o_crear_progreso_clase(db, usuario_actual.id, clase_id, vista_inicial=True)

    sena_esperada = _normalizar_sena(
        getattr(clase, "sena", None) or getattr(clase, "palabra", None) or getattr(clase, "titulo", None)
    )
    sena_detectada = _normalizar_sena(getattr(datos, "sena_reconocida", None))

    precision_cliente = max(0.0, min(1.0, _a_fraccion(datos.precision or 0, 0.0)))
    precision_minima = _a_fraccion(getattr(clase, "precision_minima", None), PRECISION_MINIMA_DEFECTO)

    sena_coincide = bool(sena_esperada) and sena_esperada == sena_detectada
    precision_valida = precision_cliente >= precision_minima
    es_exitoso = sena_coincide and precision_valida

    resultado = progreso.registrar_intento(
        precision=precision_cliente if es_exitoso else 0.0,
        duracion_segundos=max(0, int(datos.tiempo_practica or 0)),
        clase_obj=clase,
        es_exitoso=es_exitoso,
    )

    _confirmar(db)
    db.refresh(progreso)

    if es_exitoso:
        mensaje = "Práctica registrada exitosamente"
    elif not sena_coincide and not precision_valida:
        mensaje = (
            f"Seña incorrecta y precisión insuficiente "
            f"({precision_cliente * 100:.1f}% < {precision_minima * 100:.0f}%)"
        )
    elif not sena_coincide:
        mensaje = (
            f"Seña detectada '{sena_detectada or '—'}' no coincide con "
            f"la esperada '{sena_esperada or '—'}'"
        )
    else:
        mensaje = (
            f"Precisión insuficiente: {precision_cliente * 100:.1f}% "
            f"(mínimo {precision_minima * 100:.0f}%)"
        )

    return RespuestaAPI(
        exito=True,
        mensaje=mensaje,
        datos={
            "puntos_ganados": resultado["puntos_ganados"],
            "xp_ganado": resultado["xp_ganado"],
            "precision": resultado["precision"],
            "es_exitoso": es_exitoso,
            "sena_coincide": sena_coincide,
            "precision_valida": precision_valida,
            "nivel_dominio": resultado["nivel_dominio"],
            "clase_aprobada": progreso.aprobada,
            "clase_completada": progreso.completada,
        },
    )


@router.get("/clase/{clase_id}/resumen", response_model=ResumenDesempenoClase)
def obtener_resumen_desempeno_clase(
    clase_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    progreso = (
        db.query(ProgresoClase)
        .filter(ProgresoClase.usuario_id == usuario_actual.id, ProgresoClase.clase_id == clase_id)
        .first()
    )

    if not progreso:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Progreso no encontrado para esta clase",
        )

    return ResumenDesempenoClase(
        clase_id=clase_id,
        completada=progreso.completada,
        aprobada=progreso.aprobada,
        intentos_realizados=progreso.intentos_realizados,
        intentos_exitosos=progreso.intentos_exitosos,
        tasa_exito=progreso.tasa_exito,
        mejor_precision=progreso.mejor_precision,
        precision_promedio=progreso.precision_promedio,
        ultima_precision=progreso.ultima_precision,
        nivel_dominio=progreso.nivel_dominio,
        puntos_ganados=progreso.puntos_ganados,
        xp_ganado=progreso.xp_ganado,
        tiempo_total_practica_minutos=(progreso.tiempo_total_practica or 0) // 60,
        fecha_completada=progreso.fecha_completada.isoformat() if progreso.fecha_completada else None,
    )


@router.get("/leccion/{leccion_id}", response_model=ProgresoLeccionRespuesta)
def obtener_progreso_leccion(
    leccion_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    progreso = (
        db.query(ProgresoLeccion)
        .filter(ProgresoLeccion.usuario_id == usuario_actual.id, ProgresoLeccion.leccion_id == leccion_id)
        .first()
    )
    if progreso:
        return progreso

    leccion = db.query(Leccion).filter(Leccion.id == leccion_id).first()
    if not leccion:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lección no encontrada")

    total_clases = (
        db.query(func.count(Clase.id))
        .filter(Clase.leccion_id == leccion_id, Clase.activa.is_(True))
        .scalar()
    ) or 0

    progreso = ProgresoLeccion(
        usuario_id=usuario_actual.id,
        leccion_id=leccion_id,
        total_clases=total_clases,
        desbloqueada=(leccion.orden == 1),
    )
    db.add(progreso)

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        progreso = (
            db.query(ProgresoLeccion)
            .filter(ProgresoLeccion.usuario_id == usuario_actual.id, ProgresoLeccion.leccion_id == leccion_id)
            .first()
        )
        if progreso is None:
            raise
        return progreso

    db.refresh(progreso)
    return progreso


@router.get("/usuario/resumen")
def obtener_resumen_progreso_usuario(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    lecciones = db.query(Leccion).filter(Leccion.activa.is_(True)).order_by(Leccion.orden).all()
    ids = [leccion.id for leccion in lecciones]
    metricas = _metricas_lecciones(db, usuario_actual.id, ids)

    existentes = {}
    if ids:
        existentes = {
            p.leccion_id: p
            for p in db.query(ProgresoLeccion)
            .filter(ProgresoLeccion.usuario_id == usuario_actual.id, ProgresoLeccion.leccion_id.in_(ids))
            .all()
        }

    puntos_examenes_total = (
        db.query(func.coalesce(func.sum(ResultadoExamen.puntuacion_obtenida), 0))
        .filter(ResultadoExamen.usuario_id == usuario_actual.id, ResultadoExamen.aprobado.is_(True))
        .scalar()
    )

    procesadas = []
    for leccion in lecciones:
        m = metricas[leccion.id]
        progreso = existentes.get(leccion.id)

        if progreso is None:
            progreso = ProgresoLeccion(
                usuario_id=usuario_actual.id,
                leccion_id=leccion.id,
                desbloqueada=(leccion.orden == 1),
                iniciada=False,
                completada=False,
                estrellas=0,
                mejor_precision=0.0,
            )
            db.add(progreso)

        completada = _aplicar_conteos(progreso, m["total_clases"], m["clases_completadas"])

        progreso.total_puntos = m["puntos_clases"] + m["puntos_examenes"]
        progreso.xp_total = m["xp_clases"]
        progreso.total_intentos = m["total_intentos"]

        if m["precision_promedio"]:
            progreso.mejor_precision = m["precision_promedio"]
            progreso.estrellas = _estrellas(progreso.mejor_precision)

        progreso.tiene_estrella_dorada = (progreso.mejor_precision or 0) >= 0.95
        progreso.estrella_dorada = progreso.tiene_estrella_dorada
        progreso.porcentaje_precision = f"{(progreso.mejor_precision or 0) * 100:.1f}%"

        procesadas.append((leccion, progreso, m, completada))

    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Conflicto al actualizar el progreso, intenta de nuevo",
        )

    progresos_data = []
    lecciones_completadas = 0
    puntos_clases_total = 0
    total_estrellas = 0
    total_clases_global = 0
    clases_completadas_global = 0

    for leccion, progreso, m, completada in procesadas:
        total_clases_global += m["total_clases"]
        clases_completadas_global += m["clases_completadas"]
        puntos_clases_total += m["puntos_clases"]
        total_estrellas += progreso.estrellas or 0
        lecciones_completadas += 1 if completada else 0

        progresos_data.append({
            "id": progreso.id,
            "leccion_id": leccion.id,
            "leccion_titulo": leccion.titulo,
            "usuario_id": usuario_actual.id,
            "desbloqueada": progreso.desbloqueada,
            "iniciada": progreso.iniciada,
            "completada": completada,
            "total_clases": m["total_clases"],
            "clases_completadas": m["clases_completadas"],
            "mejor_precision": float(progreso.mejor_precision or 0),
            "total_intentos": progreso.total_intentos or 0,
            "total_puntos": progreso.total_puntos or 0,
            "estrellas": progreso.estrellas or 0,
            "porcentaje_completado": progreso.porcentaje_completado,
            "porcentaje_precision": progreso.porcentaje_precision,
            "tiene_estrella_dorada": progreso.tiene_estrella_dorada,
            "fecha_desbloqueo": progreso.fecha_desbloqueo,
            "fecha_completada": progreso.fecha_completada,
            "puntos_examenes": m["puntos_examenes"],
            "puntos_clases": m["puntos_clases"],
        })

    _confirmar(db)

    porcentaje_general = (
        round(clases_completadas_global / total_clases_global * 100, 1) if total_clases_global else 0
    )

    total_examenes_aprobados = (
        db.query(func.count(ResultadoExamen.id))
        .filter(ResultadoExamen.usuario_id == usuario_actual.id, ResultadoExamen.aprobado.is_(True))
        .scalar()
    ) or 0
    total_examenes = db.query(func.count(Examen.id)).filter(Examen.activo.is_(True)).scalar() or 0

    return {
        "total_lecciones": len(lecciones),
        "lecciones_completadas": lecciones_completadas,
        "porcentaje_completado": porcentaje_general,
        "total_puntos": puntos_clases_total + int(puntos_examenes_total),
        "total_estrellas": total_estrellas,
        "progresos": progresos_data,
        "estadisticas_clases": {
            "clases_completadas": clases_completadas_global,
            "total_clases": total_clases_global,
            "porcentaje_clases": porcentaje_general,
        },
        "estadisticas_examenes": {
            "total_examenes": total_examenes,
            "examenes_aprobados": total_examenes_aprobados,
            "puntos_examenes": int(puntos_examenes_total),
            "porcentaje_examenes_completados": (
                round(total_examenes_aprobados / total_examenes * 100, 1) if total_examenes else 0
            ),
        },
    }


@router.post("/actualizar-progreso")
def actualizar_progreso_lecciones(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    existentes = db.query(ProgresoLeccion).filter(ProgresoLeccion.usuario_id == usuario_actual.id).all()
    conteos = _conteos_clases(db, usuario_actual.id, [p.leccion_id for p in existentes])

    for progreso in existentes:
        c = conteos[progreso.leccion_id]
        _aplicar_conteos(progreso, c["total_clases"], c["clases_completadas"])

    _confirmar(db)

    return {
        "mensaje": "Progreso actualizado exitosamente",
        "detalle": f"Se actualizaron {len(existentes)} lecciones",
    }


@router.get("/estadisticas/gamificacion", response_model=RespuestaAPI)
def obtener_estadisticas_gamificacion(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    total_lecciones = db.query(func.count(Leccion.id)).filter(Leccion.activa.is_(True)).scalar() or 0
    total_clases = db.query(func.count(Clase.id)).filter(Clase.activa.is_(True)).scalar() or 0

    lecciones_completadas = (
        db.query(func.count(ProgresoLeccion.id))
        .filter(ProgresoLeccion.usuario_id == usuario_actual.id, ProgresoLeccion.completada.is_(True))
        .scalar()
    ) or 0

    clases_completadas, tiempo_total_segundos, precision_media = (
        db.query(
            func.coalesce(func.sum(case((ProgresoClase.completada.is_(True), 1), else_=0)), 0),
            func.coalesce(func.sum(ProgresoClase.tiempo_total_practica), 0),
            func.avg(case((ProgresoClase.mejor_precision > 0, ProgresoClase.mejor_precision))),
        )
        .filter(ProgresoClase.usuario_id == usuario_actual.id)
        .one()
    )

    puntos = _calcular_puntos_usuario(db, usuario_actual.id)

    total_examenes = db.query(func.count(Examen.id)).filter(Examen.activo.is_(True)).scalar() or 0
    examenes_aprobados = (
        db.query(func.count(ResultadoExamen.id))
        .filter(ResultadoExamen.usuario_id == usuario_actual.id, ResultadoExamen.aprobado.is_(True))
        .scalar()
    ) or 0

    xp_total = puntos["xp_total"]
    precision_global = round(float(precision_media or 0) * 100, 1)
    racha_actual = getattr(usuario_actual, "racha_actual", 0)
    mejor_racha = getattr(usuario_actual, "mejor_racha", 0)

    logros_desbloqueados = sum([
        lecciones_completadas >= 1,
        precision_global >= 80,
        racha_actual >= 7,
        examenes_aprobados >= 1,
    ])

    return RespuestaAPI(
        exito=True,
        mensaje="Estadísticas de gamificación obtenidas",
        datos={
            "puntos_totales": puntos["puntos_totales"],
            "xp_total": xp_total,
            "nivel_actual": _calcular_nivel(xp_total),
            "progreso_nivel": xp_total % XP_POR_NIVEL,
            "lecciones_completadas": lecciones_completadas,
            "total_lecciones": total_lecciones,
            "clases_completadas": int(clases_completadas),
            "total_clases": total_clases,
            "racha_actual": racha_actual,
            "mejor_racha": mejor_racha,
            "tiempo_total_practica_horas": round(tiempo_total_segundos / 3600, 1),
            "precision_global": precision_global,
            "logros_desbloqueados": logros_desbloqueados,
            "total_logros": 10,
            "rank_global": None,
            "estadisticas_examenes": {
                "total_examenes": total_examenes,
                "examenes_aprobados": examenes_aprobados,
                "puntos_examenes": puntos["puntos_examenes"],
                "porcentaje_examenes_completados": (
                    round(examenes_aprobados / total_examenes * 100, 1) if total_examenes else 0
                ),
            },
            "desglose_puntos": {
                "puntos_clases": puntos["puntos_clases"],
                "puntos_examenes": puntos["puntos_examenes"],
                "total_puntos_clases_examenes": puntos["puntos_totales"],
            },
            "desglose_xp": {"xp_clases": xp_total, "xp_total": xp_total},
        },
    )


@router.get("/ranking", response_model=RespuestaLista)
def obtener_ranking(
    limite: int = Query(10, ge=1, le=100),
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    puntos_clases_sq = (
        db.query(
            ProgresoClase.usuario_id.label("usuario_id"),
            func.sum(ProgresoClase.puntos_ganados).label("puntos_clases"),
            func.sum(ProgresoClase.xp_ganado).label("xp_total"),
        )
        .group_by(ProgresoClase.usuario_id)
        .subquery()
    )

    puntos_examenes_sq = (
        db.query(
            ResultadoExamen.usuario_id.label("usuario_id"),
            func.sum(ResultadoExamen.puntuacion_obtenida).label("puntos_examenes"),
        )
        .filter(ResultadoExamen.aprobado.is_(True))
        .group_by(ResultadoExamen.usuario_id)
        .subquery()
    )

    puntos_clases_col = func.coalesce(puntos_clases_sq.c.puntos_clases, 0)
    puntos_examenes_col = func.coalesce(puntos_examenes_sq.c.puntos_examenes, 0)
    xp_total_col = func.coalesce(puntos_clases_sq.c.xp_total, 0)
    puntos_totales_col = puntos_clases_col + puntos_examenes_col

    resultados = (
        db.query(
            Usuario.id,
            Usuario.nombre_usuario,
            puntos_clases_col.label("puntos_clases"),
            puntos_examenes_col.label("puntos_examenes"),
            xp_total_col.label("xp_total"),
        )
        .outerjoin(puntos_clases_sq, Usuario.id == puntos_clases_sq.c.usuario_id)
        .outerjoin(puntos_examenes_sq, Usuario.id == puntos_examenes_sq.c.usuario_id)
        .join(Rol, Usuario.rol_id == Rol.id)
        .filter(and_(Usuario.activo.is_(True), Rol.codigo == ROL_RANKING))
        .order_by(puntos_totales_col.desc(), Usuario.id)
        .limit(limite)
        .all()
    )

    lecciones_completadas_map = {}
    if resultados:
        lecciones_completadas_map = dict(
            db.query(ProgresoLeccion.usuario_id, func.count(ProgresoLeccion.id))
            .filter(
                ProgresoLeccion.usuario_id.in_([row.id for row in resultados]),
                ProgresoLeccion.completada.is_(True),
            )
            .group_by(ProgresoLeccion.usuario_id)
            .all()
        )

    racha_usuario = getattr(usuario_actual, "racha_actual", 0)

    ranking_data = []
    for posicion, row in enumerate(resultados, 1):
        puntos_clases = int(row.puntos_clases or 0)
        puntos_examenes = int(row.puntos_examenes or 0)
        xp_real = int(row.xp_total or 0)

        ranking_data.append({
            "usuario_id": row.id,
            "nombre_usuario": row.nombre_usuario,
            "puntos_totales": puntos_clases + puntos_examenes,
            "nivel": _calcular_nivel(xp_real),
            "lecciones_completadas": lecciones_completadas_map.get(row.id, 0),
            "racha_actual": racha_usuario if row.id == usuario_actual.id else 0,
            "posicion": posicion,
            "avatar_url": None,
            "es_usuario_actual": row.id == usuario_actual.id,
            "desglose_puntos": {
                "puntos_clases": puntos_clases,
                "puntos_examenes": puntos_examenes,
                "xp_total": xp_real,
            },
        })

    codigo_rol_actual = db.query(Rol.codigo).filter(Rol.id == usuario_actual.rol_id).scalar()
    participa_en_ranking = bool(usuario_actual.activo) and codigo_rol_actual == ROL_RANKING

    if participa_en_ranking and not any(u["usuario_id"] == usuario_actual.id for u in ranking_data):
        totales = _calcular_puntos_usuario(db, usuario_actual.id)

        lecciones_completadas_usuario = (
            db.query(func.count(ProgresoLeccion.id))
            .filter(ProgresoLeccion.usuario_id == usuario_actual.id, ProgresoLeccion.completada.is_(True))
            .scalar()
        ) or 0

        usuarios_con_mas_puntos = (
            db.query(func.count(Usuario.id))
            .select_from(Usuario)
            .join(Rol, Usuario.rol_id == Rol.id)
            .outerjoin(puntos_clases_sq, Usuario.id == puntos_clases_sq.c.usuario_id)
            .outerjoin(puntos_examenes_sq, Usuario.id == puntos_examenes_sq.c.usuario_id)
            .filter(
                Usuario.activo.is_(True),
                Rol.codigo == ROL_RANKING,
                Usuario.id != usuario_actual.id,
                puntos_totales_col > totales["puntos_totales"],
            )
            .scalar()
        ) or 0

        ranking_data.append({
            "usuario_id": usuario_actual.id,
            "nombre_usuario": usuario_actual.nombre_usuario,
            "puntos_totales": totales["puntos_totales"],
            "nivel": _calcular_nivel(totales["xp_total"]),
            "lecciones_completadas": lecciones_completadas_usuario,
            "racha_actual": racha_usuario,
            "posicion": usuarios_con_mas_puntos + 1,
            "avatar_url": None,
            "es_usuario_actual": True,
            "desglose_puntos": {
                "puntos_clases": totales["puntos_clases"],
                "puntos_examenes": totales["puntos_examenes"],
                "xp_total": totales["xp_total"],
            },
        })

    return RespuestaLista(
        exito=True,
        mensaje=f"Ranking de {len(ranking_data)} usuarios",
        datos=ranking_data,
        total=len(ranking_data),
        pagina=1,
        por_pagina=limite,
    )


@router.get("/usuario/estadisticas-detalladas", response_model=RespuestaAPI)
def obtener_estadisticas_detalladas_usuario(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    lecciones = db.query(Leccion).filter(Leccion.activa.is_(True)).order_by(Leccion.orden).all()
    metricas = _metricas_lecciones(db, usuario_actual.id, [leccion.id for leccion in lecciones])

    progreso_lecciones = []
    for leccion in lecciones:
        m = metricas[leccion.id]
        total = m["total_clases"]
        completadas = m["clases_completadas"]

        progreso_lecciones.append({
            "leccion_id": leccion.id,
            "titulo": leccion.titulo,
            "clases_completadas": completadas,
            "total_clases": total,
            "porcentaje_completado": round(completadas / total * 100, 1) if total else 0,
            "completada": total > 0 and completadas >= total,
            "mejor_precision": m["mejor_precision"],
        })

    total_practicas, practicas_exitosas, precision_media = (
        db.query(
            func.count(ProgresoClase.id),
            func.coalesce(func.sum(case((ProgresoClase.intentos_exitosos > 0, 1), else_=0)), 0),
            func.avg(case((ProgresoClase.mejor_precision > 0, ProgresoClase.mejor_precision))),
        )
        .filter(ProgresoClase.usuario_id == usuario_actual.id)
        .one()
    )

    fecha_inicio = _ahora() - timedelta(days=7)
    dia = func.date(ProgresoClase.ultima_practica)

    tiempos_por_dia = (
        db.query(dia.label("fecha"), func.sum(ProgresoClase.tiempo_total_practica).label("tiempo_total"))
        .filter(ProgresoClase.usuario_id == usuario_actual.id, ProgresoClase.ultima_practica >= fecha_inicio)
        .group_by(dia)
        .order_by(dia)
        .all()
    )

    return RespuestaAPI(
        exito=True,
        mensaje="Estadísticas detalladas obtenidas",
        datos={
            "progreso_lecciones": progreso_lecciones,
            "resumen_practicas": {
                "total_practicas": int(total_practicas),
                "practicas_exitosas": int(practicas_exitosas),
                "tasa_exito": round(int(practicas_exitosas) / int(total_practicas) * 100, 1) if total_practicas else 0,
                "precision_promedio": float(precision_media or 0),
            },
            "tiempo_practica_7_dias": [
                {"fecha": str(fecha), "tiempo_minutos": round((tiempo_total or 0) / 60, 1)}
                for fecha, tiempo_total in tiempos_por_dia
            ],
            "logros": [
                {
                    "id": 1,
                    "nombre": "Primeros Pasos",
                    "descripcion": "Completa tu primera lección",
                    "desbloqueado": any(item["completada"] for item in progreso_lecciones),
                    "fecha_desbloqueo": None,
                },
                {
                    "id": 2,
                    "nombre": "Precisión Maestra",
                    "descripcion": "Alcanza 90% de precisión en una lección",
                    "desbloqueado": any(item["mejor_precision"] >= 0.9 for item in progreso_lecciones),
                    "fecha_desbloqueo": None,
                },
            ],
        },
    )


@router.get("/debug/calculo-puntos", include_in_schema=False)
def debug_calculo_puntos(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    if not configuracion.debug:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No encontrado")

    progresos = db.query(ProgresoClase).filter(ProgresoClase.usuario_id == usuario_actual.id).all()
    totales = _calcular_puntos_usuario(db, usuario_actual.id)

    return {
        "progresos_clases": [
            {
                "clase_id": p.clase_id,
                "completada": p.completada,
                "intentos_realizados": p.intentos_realizados,
                "puntos_ganados": p.puntos_ganados,
                "xp_ganado": p.xp_ganado,
                "mejor_precision": p.mejor_precision,
            }
            for p in progresos
        ],
        "resumen": {
            "total_progresos": len(progresos),
            "puntos_acumulados": totales["puntos_clases"],
            "xp_acumulado": totales["xp_total"],
        },
        "examenes": {"puntos_examenes": totales["puntos_examenes"]},
        "totales": {
            "puntos_totales_clases": totales["puntos_clases"],
            "xp_total_clases": totales["xp_total"],
            "puntos_totales_con_examenes": totales["puntos_totales"],
        },
    }