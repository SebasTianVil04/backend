from typing import Any, Dict, Iterable, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from app.modelos.dataset import CategoriaDataset, VideoDataset
from ..utilidades.base_datos import obtener_bd
from ..dependencias.permisos import requiere_permiso
from ..modelos.usuario import Usuario
from ..modelos.categoria import Categoria
from ..modelos.entrenamiento import ModeloIA
from ..modelos.leccion import Leccion
from ..modelos.tipo_categoria import TipoCategoria
from ..esquemas.categoria import CategoriaCrear, CategoriaActualizar
from ..esquemas.respuestas import RespuestaAPI, RespuestaLista


router = APIRouter(prefix="/categorias", tags=["Categorías"])


class AsignacionModeloLote(BaseModel):
    categoria_id: int
    modelo_id: Optional[int] = None


class AsignacionesLoteRequest(BaseModel):
    asignaciones: List[AsignacionModeloLote]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dump(modelo: BaseModel, **kwargs) -> Dict[str, Any]:
    """Compatible con Pydantic v1 (.dict) y v2 (.model_dump)."""
    if hasattr(modelo, "model_dump"):
        return modelo.model_dump(**kwargs)
    return modelo.dict(**kwargs)


def _opcion_modelo():
    """Carga solo las columnas del modelo de IA que se devuelven (evita traer clases_json, etc.)."""
    return joinedload(Categoria.modelo_ia).load_only(
        ModeloIA.nombre,
        ModeloIA.activo,
        ModeloIA.accuracy,
        ModeloIA.num_clases,
        ModeloIA.tipo_modelo,
    )


def _conteo_lecciones(db: Session, ids: Optional[Iterable[int]] = None) -> Dict[int, int]:
    """
    Total de lecciones por categoría con un único GROUP BY.
    Reemplaza a joinedload(Categoria.lecciones) + len(categoria.lecciones), que traía
    todas las lecciones (multiplicando las filas del JOIN) solo para contarlas.
    """
    consulta = (
        db.query(Categoria.id, func.count(Leccion.id))
        .outerjoin(Categoria.lecciones)
        .group_by(Categoria.id)
    )

    if ids is not None:
        ids = list(ids)
        if not ids:
            return {}
        consulta = consulta.filter(Categoria.id.in_(ids))

    return {categoria_id: int(total) for categoria_id, total in consulta.all()}


def _mapa_dataset_ids(db: Session, ids: Iterable[int]) -> Dict[int, int]:
    """Categoria.id -> CategoriaDataset.id (es el id que usan los videos del dataset)."""
    ids = list(ids)
    if not ids:
        return {}

    filas = (
        db.query(CategoriaDataset.categoria_id, func.min(CategoriaDataset.id))
        .filter(CategoriaDataset.categoria_id.in_(ids))
        .group_by(CategoriaDataset.categoria_id)
        .all()
    )
    return {categoria_id: dataset_id for categoria_id, dataset_id in filas}


def _base(c: Categoria, total_lecciones: int) -> Dict[str, Any]:
    return {
        "id": c.id,
        "nombre": c.nombre,
        "tipo_id": c.tipo_id,
        "descripcion": c.descripcion,
        "icono": c.icono,
        "color": c.color,
        "orden": c.orden,
        "nivel_requerido": c.nivel_requerido,
        "activa": c.activa,
        "total_lecciones": total_lecciones,
    }


def _con_tipo(datos: Dict[str, Any], c: Categoria) -> Dict[str, Any]:
    datos["tipo_valor"] = c.tipo_rel.valor if c.tipo_rel else None
    datos["tipo_etiqueta"] = c.tipo_rel.etiqueta if c.tipo_rel else None
    return datos


def _con_fechas(datos: Dict[str, Any], c: Categoria) -> Dict[str, Any]:
    datos["fecha_creacion"] = c.fecha_creacion
    datos["fecha_actualizacion"] = c.fecha_actualizacion
    return datos


def _detalle(c: Categoria, total_lecciones: int) -> Dict[str, Any]:
    return _con_fechas(_con_tipo(_base(c, total_lecciones), c), c)


def _total_lecciones(db: Session, categoria_id: int) -> int:
    return _conteo_lecciones(db, [categoria_id]).get(categoria_id, 0)


def _obtener_o_404(db: Session, categoria_id: int) -> Categoria:
    categoria = (
        db.query(Categoria)
        .options(joinedload(Categoria.tipo_rel))
        .filter(Categoria.id == categoria_id)
        .first()
    )
    if not categoria:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Categoría con ID {categoria_id} no encontrada",
        )
    return categoria


# ---------------------------------------------------------------------------
# Listados
# ---------------------------------------------------------------------------

@router.get("/con-modelos", response_model=RespuestaLista)
def listar_categorias_con_modelos_completo(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("categorias.ver"))
):
    try:
        categorias = (
            db.query(Categoria)
            .options(_opcion_modelo())
            .filter(Categoria.activa == True)
            .order_by(Categoria.orden)
            .all()
        )

        lecciones = _conteo_lecciones(db, [c.id for c in categorias])

        categorias_data = []
        for categoria in categorias:
            datos = _base(categoria, lecciones.get(categoria.id, 0))
            modelo = categoria.modelo_ia
            datos.update({
                "modelo_ia_id": categoria.modelo_ia_id,
                "tiene_modelo": categoria.modelo_ia_id is not None,
                "modelo_nombre": modelo.nombre if modelo else None,
                "modelo_activo": modelo.activo if modelo else False,
                "modelo_accuracy": modelo.accuracy if modelo else None,
                "modelo_num_clases": modelo.num_clases if modelo else None,
                "modelo_tipo": modelo.tipo_modelo if modelo else None,
            })
            categorias_data.append(datos)

        return RespuestaLista(
            exito=True,
            mensaje=f"Se encontraron {len(categorias_data)} categorías",
            datos=categorias_data,
            total=len(categorias_data),
            pagina=1,
            por_pagina=len(categorias_data)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al listar categorías: {str(e)}"
        )


@router.get("", response_model=RespuestaLista)
def listar_categorias(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("categorias.ver")),
    activas_solo: bool = Query(True),
    tipo_id: Optional[int] = Query(None)
):
    try:
        query = db.query(Categoria).options(
            joinedload(Categoria.tipo_rel),
            _opcion_modelo(),
        )

        if activas_solo:
            query = query.filter(Categoria.activa == True)

        if tipo_id:
            query = query.filter(Categoria.tipo_id == tipo_id)

        categorias = query.order_by(Categoria.orden).all()

        ids = [c.id for c in categorias]
        lecciones = _conteo_lecciones(db, ids)
        datasets = _mapa_dataset_ids(db, ids)

        categorias_data = []
        for categoria in categorias:
            datos = _con_tipo(_base(categoria, lecciones.get(categoria.id, 0)), categoria)
            modelo = categoria.modelo_ia
            datos.update({
                "modelo_ia_id": categoria.modelo_ia_id,
                "modelo_nombre": modelo.nombre if modelo else None,
                "modelo_activo": modelo.activo if modelo else False,
                "modelo_accuracy": modelo.accuracy if modelo else None,
                "tiene_modelo": categoria.tiene_modelo_asignado,
                # id de CategoriaDataset: es el que esperan los endpoints de captura/dataset
                "dataset_id": datasets.get(categoria.id),
            })
            categorias_data.append(_con_fechas(datos, categoria))

        return RespuestaLista(
            exito=True,
            mensaje=f"Se encontraron {len(categorias_data)} categorías",
            datos=categorias_data,
            total=len(categorias_data),
            pagina=1,
            por_pagina=len(categorias_data)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al listar categorías: {str(e)}"
        )


@router.get("/{categoria_id}", response_model=RespuestaAPI)
def obtener_categoria(
    categoria_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("categorias.ver"))
):
    try:
        categoria = _obtener_o_404(db, categoria_id)

        return RespuestaAPI(
            exito=True,
            mensaje="Categoría obtenida exitosamente",
            datos=_detalle(categoria, _total_lecciones(db, categoria_id))
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener categoría: {str(e)}"
        )


# ---------------------------------------------------------------------------
# Escritura
# ---------------------------------------------------------------------------

@router.post("", response_model=RespuestaAPI)
def crear_categoria(
    categoria: CategoriaCrear,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("admin.categorias.gestionar"))
):
    try:
        tipo = db.query(TipoCategoria).filter(
            TipoCategoria.id == categoria.tipo_id
        ).first()

        if not tipo:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tipo de categoría {categoria.tipo_id} no existe"
            )

        existente = db.query(Categoria.id).filter(
            Categoria.nombre == categoria.nombre
        ).first()

        if existente:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Ya existe una categoría con el nombre '{categoria.nombre}'"
            )

        nueva_categoria = Categoria(
            nombre=categoria.nombre,
            tipo_id=categoria.tipo_id,
            descripcion=categoria.descripcion,
            icono=categoria.icono,
            color=categoria.color,
            orden=categoria.orden,
            nivel_requerido=categoria.nivel_requerido or 1,
            activa=True
        )

        db.add(nueva_categoria)
        db.flush()

        nueva_categoria_dataset = CategoriaDataset(
            nombre=nueva_categoria.nombre,
            descripcion=nueva_categoria.descripcion,
            categoria_id=nueva_categoria.id,
            activa=True,
            orden=nueva_categoria.orden
        )

        db.add(nueva_categoria_dataset)
        db.commit()
        db.refresh(nueva_categoria)
        db.refresh(nueva_categoria_dataset)

        return RespuestaAPI(
            exito=True,
            mensaje="Categoría creada exitosamente (incluye dataset)",
            datos={
                "id": nueva_categoria.id,
                "nombre": nueva_categoria.nombre,
                "tipo_id": nueva_categoria.tipo_id,
                "descripcion": nueva_categoria.descripcion,
                "icono": nueva_categoria.icono,
                "color": nueva_categoria.color,
                "orden": nueva_categoria.orden,
                "nivel_requerido": nueva_categoria.nivel_requerido,
                "activa": nueva_categoria.activa,
                "total_lecciones": 0,
                "dataset_id": nueva_categoria_dataset.id
            }
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al crear categoría: {str(e)}"
        )


@router.put("/{categoria_id}", response_model=RespuestaAPI)
def actualizar_categoria(
    categoria_id: int,
    categoria_actualizar: CategoriaActualizar,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("admin.categorias.gestionar"))
):
    try:
        categoria = _obtener_o_404(db, categoria_id)

        if categoria_actualizar.nombre and categoria_actualizar.nombre != categoria.nombre:
            existe = db.query(Categoria.id).filter(
                Categoria.nombre == categoria_actualizar.nombre,
                Categoria.id != categoria_id
            ).first()
            if existe:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Ya existe una categoría con el nombre '{categoria_actualizar.nombre}'"
                )

        if categoria_actualizar.tipo_id and categoria_actualizar.tipo_id != categoria.tipo_id:
            tipo = db.query(TipoCategoria).filter(
                TipoCategoria.id == categoria_actualizar.tipo_id
            ).first()

            if not tipo:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"El tipo de categoría con ID {categoria_actualizar.tipo_id} no existe"
                )

            if not tipo.activo:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"El tipo '{tipo.etiqueta}' está inactivo"
                )

        for campo, valor in _dump(categoria_actualizar, exclude_unset=True).items():
            setattr(categoria, campo, valor)

        categoria_dataset = db.query(CategoriaDataset).filter(
            CategoriaDataset.categoria_id == categoria_id
        ).first()

        if categoria_dataset:
            if categoria_actualizar.nombre is not None:
                categoria_dataset.nombre = categoria_actualizar.nombre
            if categoria_actualizar.descripcion is not None:
                categoria_dataset.descripcion = categoria_actualizar.descripcion
            if categoria_actualizar.activa is not None:
                categoria_dataset.activa = categoria_actualizar.activa
            if categoria_actualizar.orden is not None:
                categoria_dataset.orden = categoria_actualizar.orden

        db.commit()

        return RespuestaAPI(
            exito=True,
            mensaje=f"Categoría '{categoria.nombre}' actualizada exitosamente",
            datos=_detalle(categoria, _total_lecciones(db, categoria_id))
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al actualizar categoría: {str(e)}"
        )


@router.delete("/{categoria_id}", response_model=RespuestaAPI)
def eliminar_categoria(
    categoria_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("admin.categorias.gestionar")),
    forzar: bool = False
):
    try:
        categoria = _obtener_o_404(db, categoria_id)

        total_lecciones = _total_lecciones(db, categoria_id)

        categoria_dataset = db.query(CategoriaDataset).filter(
            CategoriaDataset.categoria_id == categoria_id
        ).first()

        total_videos = 0
        if categoria_dataset:
            total_videos = db.query(func.count(VideoDataset.id)).filter(
                VideoDataset.categoria_id == categoria_dataset.id
            ).scalar() or 0

        tiene_dependencias = total_lecciones > 0 or total_videos > 0

        if tiene_dependencias and not forzar:
            detalle = []
            if total_lecciones > 0:
                detalle.append(f"{total_lecciones} lección(es)")
            if total_videos > 0:
                detalle.append(f"{total_videos} video(s) de dataset")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"La categoría tiene {' y '.join(detalle)} asociada(s). Usa 'forzar=true' para desactivarla."
            )

        if tiene_dependencias and forzar:
            categoria.activa = False
            if categoria_dataset:
                categoria_dataset.activa = False
            db.commit()
            mensaje = "Categoría desactivada (tenía datos asociados que no se pueden borrar)"
        else:
            if categoria_dataset:
                db.delete(categoria_dataset)
            db.delete(categoria)
            db.commit()
            mensaje = "Categoría eliminada exitosamente"

        return RespuestaAPI(
            exito=True,
            mensaje=mensaje,
            datos=None
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al eliminar categoría: {str(e)}"
        )


@router.patch("/{categoria_id}/toggle", response_model=RespuestaAPI)
def cambiar_estado_categoria(
    categoria_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("admin.categorias.gestionar"))
):
    try:
        categoria = _obtener_o_404(db, categoria_id)

        categoria.activa = not categoria.activa

        categoria_dataset = db.query(CategoriaDataset).filter(
            CategoriaDataset.categoria_id == categoria_id
        ).first()

        if categoria_dataset:
            categoria_dataset.activa = categoria.activa

        db.commit()

        estado = "activada" if categoria.activa else "desactivada"
        return RespuestaAPI(
            exito=True,
            mensaje=f"Categoría '{categoria.nombre}' {estado} exitosamente",
            datos=_detalle(categoria, _total_lecciones(db, categoria_id))
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al cambiar estado de categoría: {str(e)}"
        )


@router.post("/sincronizar-dataset", response_model=RespuestaAPI)
def sincronizar_categorias_dataset(
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("admin.categorias.gestionar"))
):
    try:
        categorias = db.query(Categoria).all()

        # Una sola consulta en lugar de una por categoría
        con_dataset = {
            fila[0]
            for fila in db.query(CategoriaDataset.categoria_id)
            .filter(CategoriaDataset.categoria_id.isnot(None))
            .all()
        }

        categorias_creadas = 0
        categorias_existentes = 0

        for categoria in categorias:
            if categoria.id in con_dataset:
                categorias_existentes += 1
                continue

            db.add(CategoriaDataset(
                nombre=categoria.nombre,
                descripcion=categoria.descripcion,
                categoria_id=categoria.id,
                activa=categoria.activa,
                orden=categoria.orden
            ))
            categorias_creadas += 1

        db.commit()

        return RespuestaAPI(
            exito=True,
            mensaje=f"Sincronización completada: {categorias_creadas} creadas, {categorias_existentes} ya existían",
            datos={
                "categorias_creadas": categorias_creadas,
                "categorias_existentes": categorias_existentes,
                "total_categorias": len(categorias)
            }
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al sincronizar: {str(e)}"
        )


# ---------------------------------------------------------------------------
# Asignación de modelos
# ---------------------------------------------------------------------------

@router.patch("/{categoria_id}/asignar-modelo", response_model=RespuestaAPI)
def asignar_modelo_a_categoria(
    categoria_id: int,
    modelo_id: int = Query(..., description="ID del modelo a asignar"),
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("admin.categorias.gestionar"))
):
    try:
        categoria = db.query(Categoria).filter(Categoria.id == categoria_id).first()
        if not categoria:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Categoría {categoria_id} no encontrada"
            )

        modelo = db.query(ModeloIA).filter(ModeloIA.id == modelo_id).first()
        if not modelo:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Modelo {modelo_id} no encontrado"
            )

        modelo_anterior_id = categoria.modelo_ia_id
        categoria.modelo_ia_id = modelo_id
        db.commit()

        mensaje = f"Modelo '{modelo.nombre}' asignado a categoría '{categoria.nombre}'"
        if modelo_anterior_id:
            mensaje += f" (reemplazó modelo anterior ID: {modelo_anterior_id})"

        return RespuestaAPI(
            exito=True,
            mensaje=mensaje,
            datos={
                "categoria_id": categoria.id,
                "categoria_nombre": categoria.nombre,
                "modelo_id": modelo.id,
                "modelo_nombre": modelo.nombre,
                "modelo_accuracy": modelo.accuracy,
                "modelo_anterior_id": modelo_anterior_id
            }
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error asignando modelo: {str(e)}"
        )


@router.delete("/{categoria_id}/desasignar-modelo", response_model=RespuestaAPI)
def desasignar_modelo_de_categoria(
    categoria_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("admin.categorias.gestionar"))
):
    try:
        categoria = (
            db.query(Categoria)
            .options(_opcion_modelo())
            .filter(Categoria.id == categoria_id)
            .first()
        )
        if not categoria:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Categoría {categoria_id} no encontrada"
            )

        modelo_anterior_id = categoria.modelo_ia_id
        modelo_anterior_nombre = categoria.modelo_ia.nombre if categoria.modelo_ia else None

        categoria.modelo_ia_id = None
        db.commit()

        return RespuestaAPI(
            exito=True,
            mensaje=f"Modelo '{modelo_anterior_nombre}' desasignado de categoría '{categoria.nombre}'",
            datos={
                "categoria_id": categoria.id,
                "modelo_anterior_id": modelo_anterior_id,
                "modelo_anterior_nombre": modelo_anterior_nombre
            }
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error desasignando modelo: {str(e)}"
        )


@router.post("/asignar-modelos-lote", response_model=RespuestaAPI)
def asignar_modelos_en_lote(
    request: AsignacionesLoteRequest,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(requiere_permiso("admin.categorias.gestionar"))
):
    try:
        asignaciones = request.asignaciones

        if not asignaciones:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No se proporcionaron asignaciones"
            )

        # Precarga en 2 consultas (antes: 2-3 consultas por cada asignación)
        ids_categoria = {a.categoria_id for a in asignaciones}
        ids_modelo = {a.modelo_id for a in asignaciones if a.modelo_id is not None}

        categorias = {
            c.id: c
            for c in db.query(Categoria)
            .options(_opcion_modelo())
            .filter(Categoria.id.in_(ids_categoria))
            .all()
        }
        modelos = {
            m.id: m
            for m in db.query(ModeloIA).filter(ModeloIA.id.in_(ids_modelo)).all()
        } if ids_modelo else {}

        asignaciones_exitosas = []
        asignaciones_fallidas = []

        for asignacion in asignaciones:
            categoria_id = asignacion.categoria_id
            modelo_id = asignacion.modelo_id

            categoria = categorias.get(categoria_id)
            if not categoria:
                asignaciones_fallidas.append({
                    "categoria_id": categoria_id,
                    "error": "Categoría no encontrada"
                })
                continue

            modelo_anterior_id = categoria.modelo_ia_id
            modelo_anterior_nombre = categoria.modelo_ia.nombre if categoria.modelo_ia else None

            if modelo_id is not None:
                modelo = modelos.get(modelo_id)
                if not modelo:
                    asignaciones_fallidas.append({
                        "categoria_id": categoria_id,
                        "error": f"Modelo {modelo_id} no encontrado"
                    })
                    continue

                categoria.modelo_ia_id = modelo_id
                accion = "asignado"
                modelo_nombre = modelo.nombre
            else:
                categoria.modelo_ia_id = None
                accion = "desasignado"
                modelo_nombre = None

            asignaciones_exitosas.append({
                "categoria_id": categoria_id,
                "categoria_nombre": categoria.nombre,
                "modelo_id": modelo_id,
                "modelo_nombre": modelo_nombre,
                "modelo_anterior_id": modelo_anterior_id,
                "modelo_anterior_nombre": modelo_anterior_nombre,
                "accion": accion
            })

        db.commit()

        mensaje = f"{len(asignaciones_exitosas)} asignaciones exitosas"
        if asignaciones_fallidas:
            mensaje += f", {len(asignaciones_fallidas)} fallidas"

        return RespuestaAPI(
            exito=True,
            mensaje=mensaje,
            datos={
                "exitosas": asignaciones_exitosas,
                "fallidas": asignaciones_fallidas,
                "total_exitosas": len(asignaciones_exitosas),
                "total_fallidas": len(asignaciones_fallidas)
            }
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en asignación en lote: {str(e)}"
        )