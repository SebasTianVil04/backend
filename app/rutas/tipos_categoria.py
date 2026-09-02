# app/rutas/tipos_categoria.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict  # AGREGAR ESTO

from ..utilidades.base_datos import obtener_bd
from ..utilidades.seguridad import verificar_admin
from ..modelos.usuario import Usuario
from ..modelos.tipo_categoria import TipoCategoria
from ..modelos.categoria import Categoria
from ..esquemas.tipo_categoria_schemas import (
    TipoCategoriaCrear,
    TipoCategoriaActualizar,
    TipoCategoriaRespuesta
)
from ..esquemas.respuestas import RespuestaAPI

router = APIRouter(prefix="/tipos-categoria", tags=["Tipos de Categoría"])

@router.get("", response_model=List[Dict])  # CAMBIAR: list[dict] -> List[Dict]
async def listar_tipos_categoria(
    solo_activos: bool = True,
    db: Session = Depends(obtener_bd)
):
    """Listar todos los tipos de categoría"""
    try:
        query = db.query(TipoCategoria)
        
        if solo_activos:
            query = query.filter(TipoCategoria.activo == True)
        
        tipos = query.order_by(TipoCategoria.id).all()
        
        tipos_data = []
        for tipo in tipos:
            tipos_data.append({
                "id": tipo.id,
                "valor": tipo.valor,
                "etiqueta": tipo.etiqueta,
                "icono": tipo.icono,
                "color": tipo.color,
                "activo": tipo.activo,
                "fecha_creacion": tipo.fecha_creacion
            })
        
        return tipos_data
        
    except Exception as e:
        print(f"❌ Error al listar tipos: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al listar tipos: {str(e)}"
        )

@router.get("/{tipo_id}", response_model=RespuestaAPI)
async def obtener_tipo_categoria(
    tipo_id: int,
    db: Session = Depends(obtener_bd)
):
    """Obtener un tipo de categoría por ID"""
    try:
        tipo = db.query(TipoCategoria).filter(TipoCategoria.id == tipo_id).first()
        
        if not tipo:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tipo de categoría no encontrado"
            )
        
        tipo_data = {
            "id": tipo.id,
            "valor": tipo.valor,
            "etiqueta": tipo.etiqueta,
            "icono": tipo.icono,
            "color": tipo.color,
            "activo": tipo.activo
        }
        
        return RespuestaAPI(
            exito=True,
            mensaje="Tipo obtenido exitosamente",
            datos=tipo_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error al obtener tipo: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )

@router.post("", response_model=RespuestaAPI)
async def crear_tipo_categoria(
    datos: TipoCategoriaCrear,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    """Crear un nuevo tipo de categoría"""
    try:
        existe = db.query(TipoCategoria).filter(
            TipoCategoria.valor == datos.valor
        ).first()
        
        if existe:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Ya existe un tipo con valor '{datos.valor}'"
            )
        
        nuevo_tipo = TipoCategoria(
            valor=datos.valor,
            etiqueta=datos.etiqueta,
            icono=datos.icono,
            color=datos.color,
            activo=True
        )
        
        db.add(nuevo_tipo)
        db.commit()
        db.refresh(nuevo_tipo)
        
        tipo_data = {
            "id": nuevo_tipo.id,
            "valor": nuevo_tipo.valor,
            "etiqueta": nuevo_tipo.etiqueta,
            "icono": nuevo_tipo.icono,
            "color": nuevo_tipo.color,
            "activo": nuevo_tipo.activo
        }
        
        return RespuestaAPI(
            exito=True,
            mensaje="Tipo creado exitosamente",
            datos=tipo_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"❌ Error al crear tipo: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )

@router.put("/{tipo_id}", response_model=RespuestaAPI)
async def actualizar_tipo_categoria(
    tipo_id: int,
    datos: TipoCategoriaActualizar,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    """Actualizar un tipo de categoría"""
    try:
        tipo = db.query(TipoCategoria).filter(TipoCategoria.id == tipo_id).first()
        
        if not tipo:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tipo no encontrado"
            )
        
        datos_actualizar = datos.dict(exclude_unset=True)
        
        for campo, valor in datos_actualizar.items():
            setattr(tipo, campo, valor)
        
        db.commit()
        db.refresh(tipo)
        
        tipo_data = {
            "id": tipo.id,
            "valor": tipo.valor,
            "etiqueta": tipo.etiqueta,
            "icono": tipo.icono,
            "color": tipo.color,
            "activo": tipo.activo
        }
        
        return RespuestaAPI(
            exito=True,
            mensaje="Tipo actualizado exitosamente",
            datos=tipo_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"❌ Error al actualizar tipo: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )

@router.delete("/{tipo_id}", response_model=RespuestaAPI)
async def eliminar_tipo_categoria(
    tipo_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    """Eliminar un tipo de categoría"""
    try:
        tipo = db.query(TipoCategoria).filter(TipoCategoria.id == tipo_id).first()
        
        if not tipo:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tipo no encontrado"
            )
        
        # Proteger tipos iniciales
        tipos_protegidos = ['abecedario', 'numeros', 'saludos']
        if tipo.valor in tipos_protegidos:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No se puede eliminar tipo protegido '{tipo.etiqueta}'"
            )
        
        # Verificar que no haya categorías usándolo
        categorias = db.query(Categoria).filter(
            Categoria.tipo_id == tipo_id
        ).count()
        
        if categorias > 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Hay {categorias} categorías usando este tipo"
            )
        
        db.delete(tipo)
        db.commit()
        
        return RespuestaAPI(
            exito=True,
            mensaje="Tipo eliminado exitosamente"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"❌ Error al eliminar tipo: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )

@router.patch("/{tipo_id}/toggle", response_model=RespuestaAPI)
async def cambiar_estado_tipo(
    tipo_id: int,
    db: Session = Depends(obtener_bd),
    usuario_actual: Usuario = Depends(verificar_admin)
):
    """Activar/desactivar un tipo"""
    try:
        tipo = db.query(TipoCategoria).filter(TipoCategoria.id == tipo_id).first()
        
        if not tipo:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tipo no encontrado"
            )
        
        tipo.activo = not tipo.activo
        db.commit()
        db.refresh(tipo)
        
        estado = "activado" if tipo.activo else "desactivado"
        
        return RespuestaAPI(
            exito=True,
            mensaje=f"Tipo {estado} exitosamente"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"❌ Error al cambiar estado: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )
