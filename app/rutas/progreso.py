from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import and_, func
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
from ..modelos.usuario import Usuario
from ..modelos.rol import Rol
from ..utilidades.base_datos import obtener_bd
from ..utilidades.seguridad import obtener_usuario_actual


router = APIRouter(prefix="/progreso", tags=["Progreso"])

XP_POR_NIVEL = 100


def _calcular_nivel(xp: int) -> int:
    return max(1, (xp // XP_POR_NIVEL) + 1)


def _normalizar_sena(valor) -> str:
    if valor is None:
        return ""
    texto = str(valor).strip().lower()
    reemplazos = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "ü": "u", "ñ": "n",
    }
    for origen, destino in reemplazos.items():
        texto = texto.replace(origen, destino)
    return " ".join(texto.split())


def _obtener_o_crear_progreso_clase(
    db: Session,
    usuario_id: int,
    clase_id: int,
    vista_inicial: bool = False,
) -> ProgresoClase:
    progreso = (
        db.query(ProgresoClase)
        .filter(
            ProgresoClase.usuario_id == usuario_id,
            ProgresoClase.clase_id == clase_id,
        )
        .first()
    )
    if progreso:
        return progreso

    if not db.query(Clase.id).filter(Clase.id == clase_id).first():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Clase no encontrada",
        )

    progreso = ProgresoClase(
        usuario_id=usuario_id,
        clase_id=clase_id,
        vista=vista_inicial,
        completada=False,
        aprobada=False,
        intentos_realizados=0,
        mejor_precision=0.0,
        tiempo_total_practica=0,
    )
    db.add(progreso)
    db.flush()
    return progreso


def _calcular_puntos_usuario(db: Session, usuario_id: int) -> dict:
    puntos_clases = (
        db.query(func.sum(ProgresoClase.puntos_ganados))
        .filter(ProgresoClase.usuario_id == usuario_id)
        .scalar()
        or 0
    )
    puntos_examenes = (
        db.query(func.sum(ResultadoExamen.puntuacion_obtenida))
        .filter(
            ResultadoExamen.usuario_id == usuario_id,
            ResultadoExamen.aprobado.is_(True),
        )
        .scalar()
        or 0
    )
    xp_total = (
        db.query(func.sum(ProgresoClase.xp_ganado))
        .filter(ProgresoClase.usuario_id == usuario_id)
        .scalar()
        or 0
    )
    return {
        "puntos_clases": int(puntos_clases),
        "puntos_examenes": int(puntos_examenes),
        "puntos_totales": int(puntos_clases + puntos_examenes),
        "xp_total": int(xp_total),
    }


@router.get("/clase/{clase_id}", response_model=ProgresoClaseRespuesta)
def obtener_progreso_clase(
    clase_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    progreso = _obtener_o_crear_progreso_clase(
        db, usuario_actual.id, clase_id, vista_inicial=False
    )
    db.commit()
    db.refresh(progreso)
    return progreso


@router.post("/clase/{clase_id}/marcar-vista")
def marcar_clase_vista(
    clase_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    progreso = _obtener_o_crear_progreso_clase(
        db, usuario_actual.id, clase_id, vista_inicial=False
    )

    if not progreso.vista:
        progreso.vista = True
        progreso.fecha_primera_vista = datetime.now()

    db.commit()
    db.refresh(progreso)

    return {"mensaje": "Clase marcada como vista", "progreso": progreso}


@router.post("/clases/{clase_id}/practica", response_model=RespuestaAPI)
async def guardar_resultado_practica(
    clase_id: int,
    datos: RegistroIntentoRequest,
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
    db: Session = Depends(obtener_bd),
):
    try:
        progreso = _obtener_o_crear_progreso_clase(
            db, usuario_actual.id, clase_id, vista_inicial=True
        )

        clase = db.query(Clase).filter(Clase.id == clase_id).first()
        if not clase:
            raise HTTPException(status_code=404, detail="Clase no encontrada")

        sena_esperada = _normalizar_sena(
            getattr(clase, "sena", None)
            or getattr(clase, "palabra", None)
            or getattr(clase, "titulo", None)
        )
        sena_detectada = _normalizar_sena(getattr(datos, "sena_reconocida", None))

        try:
            precision_cliente = float(datos.precision or 0)
        except (TypeError, ValueError):
            precision_cliente = 0.0

        if precision_cliente > 1:
            precision_cliente = precision_cliente / 100.0

        try:
            precision_minima = float(getattr(clase, "precision_minima", 0.8) or 0.8)
        except (TypeError, ValueError):
            precision_minima = 0.8
        if precision_minima > 1:
            precision_minima = precision_minima / 100.0

        sena_coincide = bool(sena_esperada) and sena_esperada == sena_detectada
        precision_valida = precision_cliente >= precision_minima

        es_exitoso_servidor = sena_coincide and precision_valida

        precision_efectiva = precision_cliente if es_exitoso_servidor else 0.0

        resultado = progreso.registrar_intento(
            precision=precision_efectiva,
            duracion_segundos=datos.tiempo_practica,
            clase_obj=clase,
            es_exitoso=es_exitoso_servidor,
        )

        db.commit()
        db.refresh(progreso)

        if es_exitoso_servidor:
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

        respuesta_dict = {
            "puntos_ganados": resultado["puntos_ganados"],
            "xp_ganado": resultado["xp_ganado"],
            "precision": resultado["precision"],
            "es_exitoso": es_exitoso_servidor,
            "sena_coincide": sena_coincide,
            "precision_valida": precision_valida,
            "nivel_dominio": resultado["nivel_dominio"],
            "clase_aprobada": progreso.aprobada,
            "clase_completada": progreso.completada,
        }

        return RespuestaAPI(
            exito=True,
            mensaje=mensaje,
            datos=respuesta_dict,
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        return RespuestaAPI(
            exito=False,
            mensaje=f"Error al guardar práctica: {str(e)}",
            errores=[str(e)],
        )


@router.get("/clase/{clase_id}/resumen", response_model=ResumenDesempenoClase)
def obtener_resumen_desempeno_clase(
    clase_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    progreso = (
        db.query(ProgresoClase)
        .filter(
            ProgresoClase.usuario_id == usuario_actual.id,
            ProgresoClase.clase_id == clase_id,
        )
        .first()
    )

    if not progreso:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Progreso no encontrado para esta clase",
        )

    resumen = {
        "clase_id": clase_id,
        "completada": progreso.completada,
        "aprobada": progreso.aprobada,
        "intentos_realizados": progreso.intentos_realizados,
        "intentos_exitosos": progreso.intentos_exitosos,
        "tasa_exito": progreso.tasa_exito,
        "mejor_precision": progreso.mejor_precision,
        "precision_promedio": progreso.precision_promedio,
        "ultima_precision": progreso.ultima_precision,
        "nivel_dominio": progreso.nivel_dominio,
        "puntos_ganados": progreso.puntos_ganados,
        "xp_ganado": progreso.xp_ganado,
        "tiempo_total_practica_minutos": progreso.tiempo_total_practica // 60,
        "fecha_completada": (
            progreso.fecha_completada.isoformat()
            if progreso.fecha_completada
            else None
        ),
    }

    return ResumenDesempenoClase(**resumen)


@router.get("/leccion/{leccion_id}", response_model=ProgresoLeccionRespuesta)
def obtener_progreso_leccion(
    leccion_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    progreso = (
        db.query(ProgresoLeccion)
        .filter(
            ProgresoLeccion.usuario_id == usuario_actual.id,
            ProgresoLeccion.leccion_id == leccion_id,
        )
        .first()
    )

    if progreso:
        return progreso

    leccion = db.query(Leccion).filter(Leccion.id == leccion_id).first()
    if not leccion:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lección no encontrada",
        )

    progreso = ProgresoLeccion(
        usuario_id=usuario_actual.id,
        leccion_id=leccion_id,
        total_clases=len(leccion.clases) if leccion.clases else 0,
        desbloqueada=(leccion.orden == 1),
    )
    db.add(progreso)
    db.commit()
    db.refresh(progreso)
    return progreso


@router.get("/usuario/resumen")
def obtener_resumen_progreso_usuario(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    try:
        lecciones = db.query(Leccion).filter(Leccion.activa.is_(True)).all()

        progresos_data = []
        total_puntos = 0
        total_estrellas = 0
        lecciones_completadas = 0
        total_clases_global = 0
        clases_completadas_global = 0

        puntos_examenes = (
            db.query(func.sum(ResultadoExamen.puntuacion_obtenida))
            .filter(
                ResultadoExamen.usuario_id == usuario_actual.id,
                ResultadoExamen.aprobado.is_(True),
            )
            .scalar()
            or 0
        )

        for leccion in lecciones:
            progreso = (
                db.query(ProgresoLeccion)
                .filter(
                    ProgresoLeccion.usuario_id == usuario_actual.id,
                    ProgresoLeccion.leccion_id == leccion.id,
                )
                .first()
            )

            stats_clases = (
                db.query(
                    func.sum(ProgresoClase.puntos_ganados).label("total_puntos_clases"),
                    func.sum(ProgresoClase.xp_ganado).label("total_xp_clases"),
                    func.count(ProgresoClase.id).label("total_intentos"),
                    func.avg(ProgresoClase.mejor_precision).label("precision_promedio"),
                )
                .join(Clase)
                .filter(
                    ProgresoClase.usuario_id == usuario_actual.id,
                    Clase.leccion_id == leccion.id,
                    Clase.activa.is_(True),
                )
                .first()
            )

            clases_completadas_count = (
                db.query(ProgresoClase)
                .join(Clase)
                .filter(
                    ProgresoClase.usuario_id == usuario_actual.id,
                    Clase.leccion_id == leccion.id,
                    ProgresoClase.completada.is_(True),
                    Clase.activa.is_(True),
                )
                .count()
            )

            total_clases = (
                db.query(Clase)
                .filter(
                    Clase.leccion_id == leccion.id,
                    Clase.activa.is_(True),
                )
                .count()
            )

            total_clases_global += total_clases
            clases_completadas_global += clases_completadas_count

            puntos_examenes_leccion = (
                db.query(func.sum(ResultadoExamen.puntuacion_obtenida))
                .join(Examen)
                .filter(
                    ResultadoExamen.usuario_id == usuario_actual.id,
                    ResultadoExamen.aprobado.is_(True),
                    Examen.leccion_id == leccion.id,
                )
                .scalar()
                or 0
            )

            porcentaje_completado = (
                (clases_completadas_count / total_clases * 100)
                if total_clases > 0
                else 0
            )
            completada_actual = (
                clases_completadas_count >= total_clases and total_clases > 0
            )

            if not progreso:
                progreso = ProgresoLeccion(
                    usuario_id=usuario_actual.id,
                    leccion_id=leccion.id,
                    total_clases=total_clases,
                    clases_completadas=clases_completadas_count,
                    desbloqueada=(leccion.orden == 1),
                    iniciada=clases_completadas_count > 0,
                    completada=completada_actual,
                    mejor_precision=float(stats_clases.precision_promedio or 0),
                    total_intentos=stats_clases.total_intentos or 0,
                    total_puntos=int(
                        (stats_clases.total_puntos_clases or 0)
                        + puntos_examenes_leccion
                    ),
                    xp_total=int(stats_clases.total_xp_clases or 0),
                    estrellas=0,
                )
                db.add(progreso)
                db.flush()
            else:
                progreso.clases_completadas = clases_completadas_count
                progreso.total_clases = total_clases
                progreso.total_puntos = int(
                    (stats_clases.total_puntos_clases or 0) + puntos_examenes_leccion
                )
                progreso.xp_total = int(stats_clases.total_xp_clases or 0)
                progreso.total_intentos = stats_clases.total_intentos or 0

                if clases_completadas_count > 0 and not progreso.iniciada:
                    progreso.iniciada = True
                    progreso.fecha_inicio = datetime.now()

                if completada_actual and not progreso.completada:
                    progreso.completada = True
                    progreso.fecha_completada = datetime.now()
                elif not completada_actual and progreso.completada:
                    progreso.completada = False
                    progreso.fecha_completada = None

                if stats_clases.precision_promedio:
                    progreso.mejor_precision = float(stats_clases.precision_promedio)
                    if progreso.mejor_precision >= 0.95:
                        progreso.estrellas = 3
                    elif progreso.mejor_precision >= 0.85:
                        progreso.estrellas = 2
                    else:
                        progreso.estrellas = 1

            progresos_data.append(
                {
                    "id": progreso.id if progreso.id else None,
                    "leccion_id": leccion.id,
                    "leccion_titulo": leccion.titulo,
                    "usuario_id": usuario_actual.id,
                    "desbloqueada": progreso.desbloqueada,
                    "iniciada": progreso.iniciada,
                    "completada": completada_actual,
                    "total_clases": total_clases,
                    "clases_completadas": clases_completadas_count,
                    "mejor_precision": float(progreso.mejor_precision or 0),
                    "total_intentos": progreso.total_intentos or 0,
                    "total_puntos": progreso.total_puntos or 0,
                    "estrellas": progreso.estrellas or 0,
                    "porcentaje_completado": round(porcentaje_completado, 1),
                    "porcentaje_precision": (
                        f"{progreso.mejor_precision * 100:.1f}%"
                        if progreso.mejor_precision
                        else "0%"
                    ),
                    "tiene_estrella_dorada": (progreso.mejor_precision or 0) >= 0.95,
                    "fecha_desbloqueo": progreso.fecha_desbloqueo,
                    "fecha_completada": progreso.fecha_completada,
                    "puntos_examenes": int(puntos_examenes_leccion),
                    "puntos_clases": int(stats_clases.total_puntos_clases or 0),
                }
            )

            if completada_actual:
                lecciones_completadas += 1
            total_puntos += progreso.total_puntos or 0
            total_estrellas += progreso.estrellas or 0

        db.commit()

        total_lecciones = len(lecciones)
        porcentaje_general = (
            (clases_completadas_global / total_clases_global * 100)
            if total_clases_global > 0
            else 0
        )

        total_examenes_aprobados = (
            db.query(ResultadoExamen)
            .filter(
                ResultadoExamen.usuario_id == usuario_actual.id,
                ResultadoExamen.aprobado.is_(True),
            )
            .count()
        )

        total_examenes = db.query(Examen).filter(Examen.activo.is_(True)).count()

        return {
            "total_lecciones": total_lecciones,
            "lecciones_completadas": lecciones_completadas,
            "porcentaje_completado": round(porcentaje_general, 1),
            "total_puntos": total_puntos + int(puntos_examenes),
            "total_estrellas": total_estrellas,
            "progresos": progresos_data,
            "estadisticas_clases": {
                "clases_completadas": clases_completadas_global,
                "total_clases": total_clases_global,
                "porcentaje_clases": round(porcentaje_general, 1),
            },
            "estadisticas_examenes": {
                "total_examenes": total_examenes,
                "examenes_aprobados": total_examenes_aprobados,
                "puntos_examenes": int(puntos_examenes),
                "porcentaje_examenes_completados": (
                    round((total_examenes_aprobados / total_examenes * 100), 1)
                    if total_examenes > 0
                    else 0
                ),
            },
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener resumen: {str(e)}",
        )


@router.post("/actualizar-progreso")
def actualizar_progreso_lecciones(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    try:
        lecciones = db.query(Leccion).filter(Leccion.activa.is_(True)).all()

        for leccion in lecciones:
            progreso = (
                db.query(ProgresoLeccion)
                .filter(
                    ProgresoLeccion.usuario_id == usuario_actual.id,
                    ProgresoLeccion.leccion_id == leccion.id,
                )
                .first()
            )

            if not progreso:
                continue

            clases_completadas_count = (
                db.query(ProgresoClase)
                .join(Clase)
                .filter(
                    ProgresoClase.usuario_id == usuario_actual.id,
                    Clase.leccion_id == leccion.id,
                    ProgresoClase.completada.is_(True),
                    Clase.activa.is_(True),
                )
                .count()
            )

            total_clases = (
                db.query(Clase)
                .filter(
                    Clase.leccion_id == leccion.id,
                    Clase.activa.is_(True),
                )
                .count()
            )

            progreso.clases_completadas = clases_completadas_count
            progreso.total_clases = total_clases
            progreso.completada = (
                clases_completadas_count >= total_clases and total_clases > 0
            )

            if progreso.completada and not progreso.fecha_completada:
                progreso.fecha_completada = datetime.now()
            elif not progreso.completada:
                progreso.fecha_completada = None

        db.commit()

        return {
            "mensaje": "Progreso actualizado exitosamente",
            "detalle": f"Se actualizaron {len(lecciones)} lecciones",
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al actualizar progreso: {str(e)}",
        )


@router.get("/estadisticas/gamificacion", response_model=RespuestaAPI)
async def obtener_estadisticas_gamificacion(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    try:
        total_lecciones = db.query(Leccion).filter(Leccion.activa.is_(True)).count()

        lecciones_completadas = (
            db.query(ProgresoLeccion)
            .filter(
                ProgresoLeccion.usuario_id == usuario_actual.id,
                ProgresoLeccion.completada.is_(True),
            )
            .count()
        )

        clases_completadas = (
            db.query(ProgresoClase)
            .filter(
                ProgresoClase.usuario_id == usuario_actual.id,
                ProgresoClase.completada.is_(True),
            )
            .count()
        )

        total_clases = db.query(Clase).filter(Clase.activa.is_(True)).count()

        puntos_clases = (
            db.query(func.sum(ProgresoClase.puntos_ganados))
            .filter(ProgresoClase.usuario_id == usuario_actual.id)
            .scalar()
            or 0
        )

        xp_clases = (
            db.query(func.sum(ProgresoClase.xp_ganado))
            .filter(ProgresoClase.usuario_id == usuario_actual.id)
            .scalar()
            or 0
        )

        puntos_examenes = (
            db.query(func.sum(ResultadoExamen.puntuacion_obtenida))
            .filter(
                ResultadoExamen.usuario_id == usuario_actual.id,
                ResultadoExamen.aprobado.is_(True),
            )
            .scalar()
            or 0
        )

        puntos_totales = int(puntos_clases)
        xp_total = int(xp_clases)

        tiempo_total_segundos = (
            db.query(func.sum(ProgresoClase.tiempo_total_practica))
            .filter(ProgresoClase.usuario_id == usuario_actual.id)
            .scalar()
            or 0
        )
        tiempo_total_horas = round(tiempo_total_segundos / 3600, 1)

        precision_global_query = (
            db.query(func.avg(ProgresoClase.mejor_precision))
            .filter(
                ProgresoClase.usuario_id == usuario_actual.id,
                ProgresoClase.mejor_precision > 0,
            )
            .scalar()
        )
        precision_global = round(float(precision_global_query or 0) * 100, 1)

        total_examenes = db.query(Examen).filter(Examen.activo.is_(True)).count()
        examenes_aprobados = (
            db.query(ResultadoExamen)
            .filter(
                ResultadoExamen.usuario_id == usuario_actual.id,
                ResultadoExamen.aprobado.is_(True),
            )
            .count()
        )

        nivel_actual = _calcular_nivel(xp_total)
        progreso_nivel = xp_total % XP_POR_NIVEL

        racha_actual = getattr(usuario_actual, "racha_actual", 0)
        mejor_racha = getattr(usuario_actual, "mejor_racha", 0)

        logros_desbloqueados = 0
        total_logros = 10
        if lecciones_completadas >= 1:
            logros_desbloqueados += 1
        if precision_global >= 80:
            logros_desbloqueados += 1
        if racha_actual >= 7:
            logros_desbloqueados += 1
        if examenes_aprobados >= 1:
            logros_desbloqueados += 1

        estadisticas = {
            "puntos_totales": puntos_totales,
            "xp_total": xp_total,
            "nivel_actual": nivel_actual,
            "progreso_nivel": progreso_nivel,
            "lecciones_completadas": lecciones_completadas,
            "total_lecciones": total_lecciones,
            "clases_completadas": clases_completadas,
            "total_clases": total_clases,
            "racha_actual": racha_actual,
            "mejor_racha": mejor_racha,
            "tiempo_total_practica_horas": tiempo_total_horas,
            "precision_global": precision_global,
            "logros_desbloqueados": logros_desbloqueados,
            "total_logros": total_logros,
            "rank_global": None,
            "estadisticas_examenes": {
                "total_examenes": total_examenes,
                "examenes_aprobados": examenes_aprobados,
                "puntos_examenes": int(puntos_examenes),
                "porcentaje_examenes_completados": (
                    round((examenes_aprobados / total_examenes * 100), 1)
                    if total_examenes > 0
                    else 0
                ),
            },
            "desglose_puntos": {
                "puntos_clases": int(puntos_clases),
                "puntos_examenes": int(puntos_examenes),
                "total_puntos_clases_examenes": int(puntos_clases + puntos_examenes),
            },
            "desglose_xp": {
                "xp_clases": int(xp_clases),
                "xp_total": int(xp_total),
            },
        }

        return RespuestaAPI(
            exito=True,
            mensaje="Estadísticas de gamificación obtenidas",
            datos=estadisticas,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener estadísticas de gamificación: {str(e)}",
        )


@router.get("/ranking", response_model=RespuestaLista)
async def obtener_ranking(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
    limite: int = 10,
):
    try:
        puntos_clases_sq = (
            db.query(
                ProgresoClase.usuario_id.label("usuario_id"),
                func.sum(ProgresoClase.puntos_ganados).label("puntos_clases"),
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

        xp_clases_sq = (
            db.query(
                ProgresoClase.usuario_id.label("usuario_id"),
                func.sum(ProgresoClase.xp_ganado).label("xp_total"),
            )
            .group_by(ProgresoClase.usuario_id)
            .subquery()
        )

        puntos_clases_col = func.coalesce(puntos_clases_sq.c.puntos_clases, 0)
        puntos_examenes_col = func.coalesce(puntos_examenes_sq.c.puntos_examenes, 0)
        xp_total_col = func.coalesce(xp_clases_sq.c.xp_total, 0)
        puntos_totales_col = puntos_clases_col + puntos_examenes_col

        resultados = (
            db.query(
                Usuario.id,
                Usuario.nombres,
                Usuario.apellido_paterno,
                Usuario.apellido_materno,
                puntos_clases_col.label("puntos_clases"),
                puntos_examenes_col.label("puntos_examenes"),
                xp_total_col.label("xp_total"),
            )
            .outerjoin(puntos_clases_sq, Usuario.id == puntos_clases_sq.c.usuario_id)
            .outerjoin(
                puntos_examenes_sq, Usuario.id == puntos_examenes_sq.c.usuario_id
            )
            .outerjoin(xp_clases_sq, Usuario.id == xp_clases_sq.c.usuario_id)
            .join(Rol, Usuario.rol_id == Rol.id)
            .filter(
                and_(
                    Usuario.activo.is_(True),
                    Rol.codigo != "admin",
                )
            )
            .order_by(puntos_totales_col.desc())
            .limit(limite)
            .all()
        )

        usuario_ids = [row.id for row in resultados]
        lecciones_completadas_map = dict(
            db.query(
                ProgresoLeccion.usuario_id,
                func.count(ProgresoLeccion.id),
            )
            .filter(
                ProgresoLeccion.usuario_id.in_(usuario_ids),
                ProgresoLeccion.completada.is_(True),
            )
            .group_by(ProgresoLeccion.usuario_id)
            .all()
        )

        ranking_data = []
        for posicion, row in enumerate(resultados, 1):
            puntos_clases = int(row.puntos_clases or 0)
            puntos_examenes = int(row.puntos_examenes or 0)
            xp_real = int(row.xp_total or 0)

            ranking_data.append(
                {
                    "usuario_id": row.id,
                    "nombre_usuario": f"{row.nombres} {row.apellido_paterno}",
                    "puntos_totales": puntos_clases + puntos_examenes,
                    "nivel": _calcular_nivel(xp_real),
                    "lecciones_completadas": lecciones_completadas_map.get(row.id, 0),
                    "racha_actual": getattr(usuario_actual, "racha_actual", 0)
                    if row.id == usuario_actual.id
                    else 0,
                    "posicion": posicion,
                    "avatar_url": None,
                    "es_usuario_actual": row.id == usuario_actual.id,
                    "desglose_puntos": {
                        "puntos_clases": puntos_clases,
                        "puntos_examenes": puntos_examenes,
                        "xp_total": xp_real,
                    },
                }
            )

        if not usuario_actual.es_admin:
            usuario_en_ranking = any(
                user["usuario_id"] == usuario_actual.id for user in ranking_data
            )

            if not usuario_en_ranking:
                totales_usuario = _calcular_puntos_usuario(db, usuario_actual.id)

                lecciones_completadas_usuario = (
                    db.query(ProgresoLeccion)
                    .filter(
                        ProgresoLeccion.usuario_id == usuario_actual.id,
                        ProgresoLeccion.completada.is_(True),
                    )
                    .count()
                )

                usuarios_con_mas_puntos = (
                    db.query(Usuario.id)
                    .select_from(Usuario)
                    .join(Rol, Usuario.rol_id == Rol.id)
                    .outerjoin(
                        puntos_clases_sq,
                        Usuario.id == puntos_clases_sq.c.usuario_id,
                    )
                    .outerjoin(
                        puntos_examenes_sq,
                        Usuario.id == puntos_examenes_sq.c.usuario_id,
                    )
                    .filter(
                        and_(
                            Usuario.activo.is_(True),
                            Rol.codigo != "admin",
                            Usuario.id != usuario_actual.id,
                        )
                    )
                    .filter(
                        (
                            func.coalesce(puntos_clases_sq.c.puntos_clases, 0)
                            + func.coalesce(puntos_examenes_sq.c.puntos_examenes, 0)
                        )
                        > totales_usuario["puntos_totales"]
                    )
                    .count()
                )

                ranking_data.append(
                    {
                        "usuario_id": usuario_actual.id,
                        "nombre_usuario": (
                            f"{usuario_actual.nombres} "
                            f"{usuario_actual.apellido_paterno}"
                        ),
                        "puntos_totales": totales_usuario["puntos_totales"],
                        "nivel": _calcular_nivel(totales_usuario["xp_total"]),
                        "lecciones_completadas": lecciones_completadas_usuario,
                        "racha_actual": getattr(usuario_actual, "racha_actual", 0),
                        "posicion": usuarios_con_mas_puntos + 1,
                        "avatar_url": None,
                        "es_usuario_actual": True,
                        "desglose_puntos": {
                            "puntos_clases": totales_usuario["puntos_clases"],
                            "puntos_examenes": totales_usuario["puntos_examenes"],
                            "xp_total": totales_usuario["xp_total"],
                        },
                    }
                )

        return RespuestaLista(
            exito=True,
            mensaje=f"Ranking de {len(ranking_data)} usuarios (excluyendo administradores)",
            datos=ranking_data,
            total=len(ranking_data),
            pagina=1,
            por_pagina=limite,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener ranking: {str(e)}",
        )

@router.get("/usuario/estadisticas-detalladas", response_model=RespuestaAPI)
async def obtener_estadisticas_detalladas_usuario(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    try:
        lecciones = db.query(Leccion).filter(Leccion.activa.is_(True)).all()

        progreso_lecciones = []
        for leccion in lecciones:
            clases_completadas = (
                db.query(ProgresoClase)
                .join(Clase)
                .filter(
                    ProgresoClase.usuario_id == usuario_actual.id,
                    Clase.leccion_id == leccion.id,
                    ProgresoClase.completada.is_(True),
                )
                .count()
            )

            total_clases = (
                db.query(Clase)
                .filter(
                    Clase.leccion_id == leccion.id,
                    Clase.activa.is_(True),
                )
                .count()
            )

            mejor_precision = (
                db.query(func.max(ProgresoClase.mejor_precision))
                .join(Clase)
                .filter(
                    ProgresoClase.usuario_id == usuario_actual.id,
                    Clase.leccion_id == leccion.id,
                )
                .scalar()
                or 0
            )

            progreso_lecciones.append(
                {
                    "leccion_id": leccion.id,
                    "titulo": leccion.titulo,
                    "clases_completadas": clases_completadas,
                    "total_clases": total_clases,
                    "porcentaje_completado": (
                        round((clases_completadas / total_clases * 100), 1)
                        if total_clases > 0
                        else 0
                    ),
                    "completada": clases_completadas >= total_clases and total_clases > 0,
                    "mejor_precision": float(mejor_precision),
                }
            )

        total_practicas = (
            db.query(ProgresoClase)
            .filter(ProgresoClase.usuario_id == usuario_actual.id)
            .count()
        )

        practicas_exitosas = (
            db.query(ProgresoClase)
            .filter(
                ProgresoClase.usuario_id == usuario_actual.id,
                ProgresoClase.intentos_exitosos > 0,
            )
            .count()
        )

        tasa_exito = (
            round((practicas_exitosas / total_practicas * 100), 1)
            if total_practicas > 0
            else 0
        )

        fecha_inicio = datetime.now() - timedelta(days=7)

        tiempos_por_dia = (
            db.query(
                func.date(ProgresoClase.ultima_practica).label("fecha"),
                func.sum(ProgresoClase.tiempo_total_practica).label("tiempo_total"),
            )
            .filter(
                ProgresoClase.usuario_id == usuario_actual.id,
                ProgresoClase.ultima_practica >= fecha_inicio,
            )
            .group_by(func.date(ProgresoClase.ultima_practica))
            .order_by(func.date(ProgresoClase.ultima_practica))
            .all()
        )

        tiempo_practica_7_dias = [
            {
                "fecha": fecha.strftime("%Y-%m-%d"),
                "tiempo_minutos": round(tiempo_total / 60, 1),
            }
            for fecha, tiempo_total in tiempos_por_dia
        ]

        precision_promedio = (
            db.query(func.avg(ProgresoClase.mejor_precision))
            .filter(
                ProgresoClase.usuario_id == usuario_actual.id,
                ProgresoClase.mejor_precision > 0,
            )
            .scalar()
            or 0
        )

        estadisticas_detalladas = {
            "progreso_lecciones": progreso_lecciones,
            "resumen_practicas": {
                "total_practicas": total_practicas,
                "practicas_exitosas": practicas_exitosas,
                "tasa_exito": tasa_exito,
                "precision_promedio": float(precision_promedio),
            },
            "tiempo_practica_7_dias": tiempo_practica_7_dias,
            "logros": [
                {
                    "id": 1,
                    "nombre": "Primeros Pasos",
                    "descripcion": "Completa tu primera lección",
                    "desbloqueado": any(
                        leccion["completada"] for leccion in progreso_lecciones
                    ),
                    "fecha_desbloqueo": None,
                },
                {
                    "id": 2,
                    "nombre": "Precisión Maestra",
                    "descripcion": "Alcanza 90% de precisión en una lección",
                    "desbloqueado": any(
                        leccion.get("mejor_precision", 0) >= 0.9
                        for leccion in progreso_lecciones
                    ),
                    "fecha_desbloqueo": None,
                },
            ],
        }

        return RespuestaAPI(
            exito=True,
            mensaje="Estadísticas detalladas obtenidas",
            datos=estadisticas_detalladas,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener estadísticas detalladas: {str(e)}",
        )


@router.get("/debug/calculo-puntos")
async def debug_calculo_puntos(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    progresos_clases = (
        db.query(ProgresoClase)
        .filter(ProgresoClase.usuario_id == usuario_actual.id)
        .all()
    )

    debug_info = {
        "progresos_clases": [],
        "resumen": {
            "total_progresos": len(progresos_clases),
            "puntos_acumulados": 0,
            "xp_acumulado": 0,
        },
    }

    for progreso in progresos_clases:
        debug_info["progresos_clases"].append(
            {
                "clase_id": progreso.clase_id,
                "completada": progreso.completada,
                "intentos_realizados": progreso.intentos_realizados,
                "puntos_ganados": progreso.puntos_ganados,
                "xp_ganado": progreso.xp_ganado,
                "mejor_precision": progreso.mejor_precision,
            }
        )
        debug_info["resumen"]["puntos_acumulados"] += progreso.puntos_ganados
        debug_info["resumen"]["xp_acumulado"] += progreso.xp_ganado

    puntos_examenes = (
        db.query(func.sum(ResultadoExamen.puntuacion_obtenida))
        .filter(
            ResultadoExamen.usuario_id == usuario_actual.id,
            ResultadoExamen.aprobado.is_(True),
        )
        .scalar()
        or 0
    )

    debug_info["examenes"] = {"puntos_examenes": int(puntos_examenes)}
    debug_info["totales"] = {
        "puntos_totales_clases": debug_info["resumen"]["puntos_acumulados"],
        "xp_total_clases": debug_info["resumen"]["xp_acumulado"],
        "puntos_totales_con_examenes": (
            debug_info["resumen"]["puntos_acumulados"] + puntos_examenes
        ),
    }

    return debug_info