from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload, selectinload

from ..dependencias.permisos import requiere_permiso
from ..esquemas.leccion_schemas import LeccionActualizar, LeccionCrear, RespuestaAPI, RespuestaLista
from ..modelos.categoria import Categoria
from ..modelos.clase import Clase
from ..modelos.examen import Examen, ResultadoExamen
from ..modelos.leccion import Leccion
from ..modelos.progreso import ProgresoClase
from ..modelos.usuario import Usuario
from ..utilidades.base_datos import obtener_bd
from ..utilidades.seguridad import obtener_usuario_actual

router = APIRouter(prefix="/lecciones", tags=["Lecciones"])

PERMISO_GESTION = "admin.lecciones.gestionar"

CAMPOS_LECCION = (
    "id", "titulo", "descripcion", "sena", "categoria_id", "categoria_nombre",
    "orden", "nivel_dificultad", "activa", "bloqueada", "puntos_base",
    "puntos_perfecto", "leccion_previa_id", "total_clases", "fecha_creacion",
)

CAMPOS_LECCION_OPCIONALES = (
    "color_tema", "imagen_miniatura", "requiere_examen_nivel", "numero_examen",
    "nivel_dificultad_texto", "fecha_actualizacion",
)


def _confirmar(db: Session) -> None:
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise


def _puede_gestionar(usuario: Usuario) -> bool:
    return bool(usuario.es_admin or usuario.tiene_permiso(PERMISO_GESTION))


def _serializar(leccion: Leccion, total_clases: Optional[int] = None) -> dict:
    datos = {}
    for campo in CAMPOS_LECCION:
        if campo == "total_clases" and total_clases is not None:
            datos[campo] = total_clases
        else:
            datos[campo] = getattr(leccion, campo)
    for campo in CAMPOS_LECCION_OPCIONALES:
        if hasattr(leccion, campo):
            datos[campo] = getattr(leccion, campo)
    return datos


def _conteo_clases_activas(db: Session, leccion_ids: list[int]) -> dict[int, int]:
    if not leccion_ids:
        return {}
    filas = (
        db.query(Clase.leccion_id, func.count(Clase.id))
        .filter(Clase.leccion_id.in_(leccion_ids), Clase.activa.is_(True))
        .group_by(Clase.leccion_id)
        .all()
    )
    return {leccion_id: total for leccion_id, total in filas}


def _obtener_leccion_o_404(db: Session, leccion_id: int, solo_activas: bool = False) -> Leccion:
    query = (
        db.query(Leccion)
        .options(joinedload(Leccion.categoria_rel))
        .filter(Leccion.id == leccion_id)
    )
    if solo_activas:
        query = query.filter(Leccion.activa.is_(True))

    leccion = query.first()
    if not leccion:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lección no encontrada")
    return leccion


def _validar_categoria(db: Session, categoria_id: int) -> None:
    if not db.query(Categoria.id).filter(Categoria.id == categoria_id).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Categoría con ID {categoria_id} no existe",
        )


def _validar_leccion_previa(db: Session, previa_id: Optional[int], propio_id: Optional[int] = None) -> None:
    if previa_id is None:
        return
    if propio_id is not None and previa_id == propio_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Una lección no puede ser su propia lección previa",
        )
    if not db.query(Leccion.id).filter(Leccion.id == previa_id).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Lección previa con ID {previa_id} no existe",
        )


def _examen_disponible(examen: Examen, clases_completadas: int, total_clases: int) -> bool:
    if examen.requiere_todas_clases:
        return clases_completadas >= total_clases
    if examen.clases_requeridas > 0:
        return clases_completadas >= examen.clases_requeridas
    return True


@router.post("", response_model=RespuestaAPI)
def crear_leccion(
    leccion: LeccionCrear,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso(PERMISO_GESTION)),
):
    _validar_categoria(db, leccion.categoria_id)
    _validar_leccion_previa(db, leccion.leccion_previa_id)

    nueva_leccion = Leccion(**leccion.model_dump())
    db.add(nueva_leccion)
    _confirmar(db)
    db.refresh(nueva_leccion)

    return RespuestaAPI(exito=True, mensaje="Lección creada exitosamente", datos=_serializar(nueva_leccion))


@router.get("", response_model=RespuestaLista)
def listar_lecciones(
    categoria_id: Optional[int] = None,
    activa: Optional[bool] = None,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    query = db.query(Leccion).options(joinedload(Leccion.categoria_rel))

    if not _puede_gestionar(usuario_actual):
        activa = True

    if activa is not None:
        query = query.filter(Leccion.activa == activa)

    if categoria_id:
        query = query.filter(Leccion.categoria_id == categoria_id)

    lecciones = query.order_by(Leccion.orden).all()
    conteos = _conteo_clases_activas(db, [leccion.id for leccion in lecciones])
    datos = [_serializar(leccion, conteos.get(leccion.id, 0)) for leccion in lecciones]

    return RespuestaLista(
        exito=True,
        mensaje=f"Se encontraron {len(datos)} lecciones",
        datos=datos,
        total=len(datos),
        pagina=1,
        por_pagina=len(datos),
    )


@router.get("/{leccion_id}", response_model=RespuestaAPI)
def obtener_leccion(
    leccion_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    leccion = _obtener_leccion_o_404(db, leccion_id, solo_activas=not _puede_gestionar(usuario_actual))
    total = _conteo_clases_activas(db, [leccion.id]).get(leccion.id, 0)
    return RespuestaAPI(exito=True, mensaje="Lección obtenida", datos=_serializar(leccion, total))


@router.get("/{leccion_id}/progreso-clases", response_model=RespuestaAPI)
def obtener_progreso_clases_leccion(
    leccion_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    _obtener_leccion_o_404(db, leccion_id, solo_activas=not _puede_gestionar(usuario_actual))

    ids_clases = [
        fila.id
        for fila in (
            db.query(Clase.id)
            .filter(Clase.leccion_id == leccion_id, Clase.activa.is_(True))
            .order_by(Clase.orden)
            .all()
        )
    ]

    completadas = set()
    if ids_clases:
        completadas = {
            fila.clase_id
            for fila in (
                db.query(ProgresoClase.clase_id)
                .filter(
                    ProgresoClase.usuario_id == usuario_actual.id,
                    ProgresoClase.completada.is_(True),
                    ProgresoClase.clase_id.in_(ids_clases),
                )
                .all()
            )
        }

    clases = [{"clase_id": clase_id, "completada": clase_id in completadas} for clase_id in ids_clases]

    return RespuestaAPI(
        exito=True,
        mensaje="Progreso de clases obtenido",
        datos={
            "leccion_id": leccion_id,
            "total_clases": len(ids_clases),
            "clases_completadas": len(completadas),
            "clases": clases,
        },
    )


@router.patch("/{leccion_id}", response_model=RespuestaAPI)
def actualizar_leccion(
    leccion_id: int,
    datos: LeccionActualizar,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso(PERMISO_GESTION)),
):
    leccion = _obtener_leccion_o_404(db, leccion_id)
    cambios = datos.model_dump(exclude_unset=True)

    if cambios.get("categoria_id") is not None:
        _validar_categoria(db, cambios["categoria_id"])

    if "leccion_previa_id" in cambios:
        _validar_leccion_previa(db, cambios["leccion_previa_id"], propio_id=leccion_id)

    for campo, valor in cambios.items():
        setattr(leccion, campo, valor)

    _confirmar(db)
    db.refresh(leccion)

    total = _conteo_clases_activas(db, [leccion.id]).get(leccion.id, 0)
    return RespuestaAPI(exito=True, mensaje="Lección actualizada exitosamente", datos=_serializar(leccion, total))


@router.delete("/{leccion_id}", response_model=RespuestaAPI)
def eliminar_leccion(
    leccion_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso(PERMISO_GESTION)),
):
    leccion = _obtener_leccion_o_404(db, leccion_id)

    db.delete(leccion)
    _confirmar(db)

    return RespuestaAPI(exito=True, mensaje="Lección eliminada exitosamente", datos=None)


@router.get("/{leccion_id}/examenes", response_model=RespuestaLista)
def obtener_examenes_leccion(
    leccion_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(obtener_usuario_actual),
):
    leccion = _obtener_leccion_o_404(db, leccion_id, solo_activas=True)

    modelo_pregunta = Examen.preguntas.property.mapper.class_

    examenes = (
        db.query(Examen)
        .options(selectinload(Examen.preguntas).load_only(modelo_pregunta.id).lazyload("*"))
        .filter(Examen.leccion_id == leccion_id, Examen.activo.is_(True))
        .order_by(Examen.orden)
        .all()
    )

    clases_completadas = (
        db.query(func.count())
        .select_from(ProgresoClase)
        .join(Clase)
        .filter(
            ProgresoClase.usuario_id == usuario_actual.id,
            Clase.leccion_id == leccion_id,
            ProgresoClase.completada.is_(True),
        )
        .scalar()
    ) or 0

    total_clases = (
        db.query(func.count(Clase.id))
        .filter(Clase.leccion_id == leccion_id, Clase.activa.is_(True))
        .scalar()
    ) or 0

    mejores = {}
    if examenes:
        resultados = (
            db.query(ResultadoExamen)
            .filter(
                ResultadoExamen.usuario_id == usuario_actual.id,
                ResultadoExamen.examen_id.in_([examen.id for examen in examenes]),
            )
            .order_by(ResultadoExamen.porcentaje.desc())
            .all()
        )
        for resultado in resultados:
            mejores.setdefault(resultado.examen_id, resultado)

    datos = []
    for examen in examenes:
        mejor = mejores.get(examen.id)
        datos.append({
            "id": examen.id,
            "titulo": examen.titulo,
            "descripcion": examen.descripcion,
            "tipo": examen.tipo,
            "nivel": examen.nivel,
            "leccion_id": examen.leccion_id,
            "orden": examen.orden,
            "clases_requeridas": examen.clases_requeridas,
            "requiere_todas_clases": examen.requiere_todas_clases,
            "tiempo_limite": examen.tiempo_limite,
            "puntuacion_minima": examen.puntuacion_minima,
            "activo": examen.activo,
            "total_preguntas": len(examen.preguntas),
            "completado": mejor is not None,
            "mejor_calificacion": mejor.porcentaje if mejor else 0,
            "aprobado": mejor.aprobado if mejor else False,
            "disponible": _examen_disponible(examen, clases_completadas, total_clases),
            "fecha_creacion": examen.fecha_creacion.isoformat() if examen.fecha_creacion else None,
        })

    return RespuestaLista(
        exito=True,
        mensaje=f"Exámenes de la lección '{leccion.titulo}'",
        datos=datos,
        total=len(datos),
        pagina=1,
        por_pagina=len(datos),
    )