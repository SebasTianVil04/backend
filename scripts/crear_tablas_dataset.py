# script_inicializar_db.py
from sqlalchemy import create_engine, inspect
from app.utilidades.base_datos import Base, SessionLocal
from app.utilidades.configuracion import configuracion

from app.modelos.tipo_categoria import TipoCategoria
from app.modelos.categoria import Categoria


def crear_tablas():
    """Crear todas las tablas en la base de datos"""
    print("Conectando a la base de datos...")
    engine = create_engine(configuracion.database_url)
    
    print("Creando tablas...")
    Base.metadata.create_all(bind=engine)
    
    print("Tablas creadas exitosamente!")
    
    inspector = inspect(engine)
    tablas = inspector.get_table_names()
    
    print(f"\nTotal de tablas: {len(tablas)}")
    print("Tablas en la base de datos:")
    for tabla in sorted(tablas):
        print(f"  - {tabla}")


def inicializar_tipos_categoria(db):
    """Crear los 3 tipos de categoría iniciales"""
    tipos_iniciales = [
        {
            "valor": "abecedario",
            "etiqueta": "Abecedario",
            "icono": "🔤",
            "color": "#3B82F6"
        },
        {
            "valor": "numeros",
            "etiqueta": "Números",
            "icono": "🔢",
            "color": "#10B981"
        },
        {
            "valor": "saludos",
            "etiqueta": "Saludos",
            "icono": "👋",
            "color": "#F59E0B"
        }
    ]
    
    print("\nCreando tipos de categoría...")
    tipos_creados = {}
    
    for tipo_data in tipos_iniciales:
        existe = db.query(TipoCategoria).filter(
            TipoCategoria.valor == tipo_data["valor"]
        ).first()
        
        if not existe:
            tipo = TipoCategoria(**tipo_data, activo=True)
            db.add(tipo)
            db.flush()  # Para obtener el ID
            tipos_creados[tipo_data["valor"]] = tipo
            print(f"  ✓ Tipo creado: {tipo_data['etiqueta']} (ID: {tipo.id})")
        else:
            tipos_creados[tipo_data["valor"]] = existe
            print(f"  ✓ Tipo ya existe: {tipo_data['etiqueta']} (ID: {existe.id})")
    
    db.commit()
    print("✅ Tipos de categoría inicializados\n")
    
    return tipos_creados


def inicializar_categorias(db, tipos_creados):
    """Crear las 3 categorías iniciales"""
    categorias_iniciales = [
        {
            "nombre": "Abecedario",
            "tipo_valor": "abecedario",
            "descripcion": "Aprende las letras del abecedario en Lengua de Señas Peruana",
            "icono": "🔤",
            "color": "#3B82F6",
            "orden": 1,
            "nivel_requerido": 1,
            "activa": True
        },
        {
            "nombre": "Números",
            "tipo_valor": "numeros",
            "descripcion": "Aprende los números del 0 al 9 en Lengua de Señas Peruana",
            "icono": "🔢",
            "color": "#10B981",
            "orden": 2,
            "nivel_requerido": 1,
            "activa": True
        },
        {
            "nombre": "Saludos",
            "tipo_valor": "saludos",
            "descripcion": "Aprende saludos y despedidas básicas en Lengua de Señas Peruana",
            "icono": "👋",
            "color": "#F59E0B",
            "orden": 3,
            "nivel_requerido": 1,
            "activa": True
        }
    ]
    
    print("Creando categorías de lecciones...")
    
    for cat_data in categorias_iniciales:
        existe = db.query(Categoria).filter(
            Categoria.nombre == cat_data["nombre"]
        ).first()
        
        if not existe:
            # Obtener el tipo_id del tipo correspondiente
            tipo_valor = cat_data.pop("tipo_valor")
            
            if tipo_valor not in tipos_creados:
                print(f"  ✗ Error: Tipo '{tipo_valor}' no encontrado")
                continue
            
            tipo_id = tipos_creados[tipo_valor].id
            
            categoria = Categoria(
                **cat_data,
                tipo_id=tipo_id
            )
            db.add(categoria)
            db.flush()
            print(f"  ✓ Categoría creada: {cat_data['nombre']} → Tipo ID: {tipo_id}")
        else:
            print(f"  ✓ Categoría ya existe: {cat_data['nombre']}")
    
    db.commit()
    print("✅ Categorías de lecciones inicializadas\n")


def main():
    """Función principal"""
    print("=" * 70)
    print("INICIALIZAR BASE DE DATOS - SIGNAFREE")
    print("=" * 70)
    
    # Crear tablas
    crear_tablas()
    
    # Crear sesión
    db = SessionLocal()
    
    try:
        print("\n" + "=" * 70)
        print("INICIALIZANDO DATOS INICIALES")
        print("=" * 70)
        
        # Crear tipos de categoría primero
        tipos_creados = inicializar_tipos_categoria(db)
        
        # Crear categorías después (con referencia a tipos)
        inicializar_categorias(db, tipos_creados)
        
        # Verificar integridad
        print("Verificando integridad de relaciones...")
        categorias = db.query(Categoria).all()
        for cat in categorias:
            print(f"  • {cat.nombre} → Tipo: {cat.tipo_rel.etiqueta if cat.tipo_rel else 'SIN TIPO'}")
        
        print("\n" + "=" * 70)
        print("✅ BASE DE DATOS CONFIGURADA EXITOSAMENTE")
        print("=" * 70)
        print("\nDatos creados:")
        print("  • 3 Tipos de categoría")
        print("  • 3 Categorías de lecciones")
        print("  • Relaciones correctamente vinculadas")
        print("\nAhora puedes:")
        print("  1. Crear lecciones")
        print("  2. Crear clases")
        print("  3. Empezar a usar la aplicación")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    main()
