from services.vision_service.app.detection.colors import (
    CLASS_COLORS,
    DEFAULT_COLOR,
    FALLBACK_PALETTE,
    get_class_color,
    get_text_color,
)


class TestClassColors:

    def test_paletas_definidas_y_no_vacias(self):
        """Comprueba que las paletas globales estén definidas y pobladas."""
        assert isinstance(CLASS_COLORS, dict)
        assert len(CLASS_COLORS) > 0
        assert isinstance(FALLBACK_PALETTE, list)
        assert len(FALLBACK_PALETTE) > 0

    def test_clases_principales_tienen_colores_distintos(self):
        """Verifica que cada clase de vehículo y actor vial tenga un color único."""
        classes = [
            "car",
            "motorcycle",
            "truck",
            "bus",
            "ambulance",
            "ciclist",
            "monopatin",
            "pedestrian",
        ]

        colors = [get_class_color(cls) for cls in classes]

        # Verificar que todos los colores sean tuplas BGR válidas (3 enteros entre 0 y 255)
        for color in colors:
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(0 <= channel <= 255 for channel in color)

        # Todos deben ser distintos entre sí
        assert len(set(colors)) == len(classes), (
            f"Hay colores repetidos entre las clases: {dict(zip(classes, colors))}"
        )

    def test_sinonimos_en_espanol_retornan_mismo_color(self):
        """Comprueba que los nombres en español e inglés mapeen al mismo color."""
        pairs = [
            ("car", "carro"),
            ("motorcycle", "moto"),
            ("truck", "camion"),
            ("truck", "camión"),
            ("ambulance", "ambulancia"),
            ("ciclist", "ciclista"),
            ("monopatin", "patineta"),
            ("pedestrian", "peaton"),
            ("pedestrian", "peatón"),
        ]

        for eng, esp in pairs:
            c_eng = get_class_color(eng)
            c_esp = get_class_color(esp)
            assert c_eng == c_esp, (
                f"Color para '{eng}' ({c_eng}) no coincide con '{esp}' ({c_esp})"
            )

    def test_normalizacion_de_casing_y_espacios(self):
        """Comprueba tolerancia a mayúsculas, espacios y tildes."""
        assert get_class_color("  CAR  ") == get_class_color("car")
        assert get_class_color("Camión") == get_class_color("truck")
        assert get_class_color("MOTO") == get_class_color("motorcycle")

    def test_clases_desconocidas_reciben_color_consistente_y_valido(self):
        """Una clase desconocida debe recibir un color de la paleta y ser determinista."""
        color1 = get_class_color("helicoptero")
        color2 = get_class_color("helicoptero")

        assert color1 == color2
        assert isinstance(color1, tuple)
        assert len(color1) == 3

    def test_none_o_vacio_retorna_default_color(self):
        assert get_class_color(None) == DEFAULT_COLOR
        assert get_class_color("") == DEFAULT_COLOR
        assert get_class_color("   ") == DEFAULT_COLOR

    def test_get_text_color_contraste(self):
        """Texto negro para fondos claros y blanco para fondos oscuros."""
        # Blanco o amarillo brillante -> texto negro
        assert get_text_color((255, 255, 255)) == (0, 0, 0)
        assert get_text_color((0, 255, 255)) == (0, 0, 0)

        # Negro o azul oscuro -> texto blanco
        assert get_text_color((0, 0, 0)) == (255, 255, 255)
        assert get_text_color((255, 0, 0)) == (255, 255, 255)
