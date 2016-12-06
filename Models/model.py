from PyQt5.QtGui import QColor

from Utility.structures import TransformationMatrix, Polygon
import math


class Model:
    def __init__(self, width, height, u_step, v_step, shape, observer):
        """
        :param u_step: шаг сетки по u
        :param v_step: шаг сетки по v
        :param shape: отрисовываемая фигура
        :param observer: положение наблюдатеся в однородных координатах
        """
        self.shape = shape
        self.u_step = u_step
        self.v_step = v_step

        self.observer = observer

        self.width = width
        self.height = height

        self.points = []
        self.polygons = []

        self.transformation_matrix = self.projection_matrix(observer)

        self.light_source = observer

        self.front_color = QColor(255, 255, 0)
        self.inner_color = QColor(0, 255, 255)

    def set_shape(self, shape):
        self.shape = shape

    def set_size(self, width, height):
        self.width = width
        self.height = height

    def set_front_color(self, front_color):
        self.front_color = front_color

    def set_inner_color(self, inner_color):
        self.inner_color = inner_color

    def set_steps(self, u_step, v_step):
        self.u_step = u_step
        self.v_step = v_step
        self.shape.set_steps(u_step, v_step)

    def set_observer(self, observer):
        self.transformation_matrix = self.projection_matrix(observer)

    def set_light_source(self, light_source):
        self.light_source = light_source

    def polygon_approximation(self):
        """
        Аппроксимация полигонами.
        """
        self.points = []
        self.polygons = []

        for i in range(self.u_step):
            self.points.append([])
            for j in range(self.v_step):
                # Проективное преобразование точки.
                point = self.shape.homogeneous_point(i, j) * self.transformation_matrix
                self.points[i].append(point)
                # Формирование полигонов.
                if (i != 0)  and (j != 0):
                    self.polygons.append(Polygon(self.points[i-1][j-1], self.points[i-1][j], self.points[i][j-1]))
                    self.polygons.append(Polygon(self.points[i][j-1], self.points[i-1][j], self.points[i][j]))

    def rotation_matrix(self, observer):
        """
        Создание матрицы поворотов.
        :param observer: Положение наблюдателя в однородных координатах.
        :return:
        """
        oz_angle = math.atan2(observer.x, observer.y)
        ox_angle = math.atan2(math.sqrt(observer.x*observer.x + observer.y*observer.y), observer.z)
        return TransformationMatrix.rotation_oz(oz_angle)*TransformationMatrix.rotation_ox(ox_angle)

    def projection_matrix(self, observer):
        """
        Формирование матрицы проецирования.
        :param observer: Положения наблюдателя в однородных координатах.
        :return:
        """
        mtx = self.rotation_matrix(observer)

        # Отражение относительно плоскости YOZ
        mtx = mtx * TransformationMatrix.reflection_yoz()

        """
        # При центральном проецировании
        if self.mode == "central":
            # Координата наблюдателя: длина радиус-вектора точки наблюдателя
            observer_z = math.sqrt(observer.x * observer.x + observer.y * observer.y + observer.z * observer.z)
            # Центральное преобразование для плоскости XOY
            mtx = mtx * TransformationMatrix.central_xoy(observer_z)
        """

        # Ортогональное проецирование на плоскость XOY (плоскость экрана)
        # mtx = mtx * TransformationMatrix.ortogonal_projection_xoy()
        # Перенос в начало координата экрана.

        mtx = mtx * TransformationMatrix.transfer(self.width/2, self.height/2, 0)

        return mtx

    def wireframe_model(self):
        """
        Каркасная модель.
        :return: Список спроецированных на экран точек-вершин полигонов в виде ((A, B, C), None)
        """

        data = []

        for polygon in self.polygons:
            pol_coord = polygon.convertion_two_dim()

            data.append((pol_coord, None))

        return data

    def surface_normal(self, pol):
        """
        Вычисление нормали многоугольника.
        Задача сводится к вычислению векторного произведения векторов AB и AC.
        :param pol: Полигон
        :return: Кортеж из трех элементов с координатами вектора-нормали.
        """
        v1 = (pol.b.x - pol.a.x,
              pol.b.y - pol.a.y,
              pol.b.z - pol.a.z)

        v2 = (pol.c.x - pol.a.x,
              pol.c.y - pol.a.y,
              pol.c.z - pol.a.z)

        norm = (v1[1] * v2[2] - v1[2] * v2[1],
                v1[2] * v2[0] - v1[0] * v2[2],
                v1[0] * v2[1] - v1[1] * v2[0])

        return norm

    def flat_shading(self):
        """
        Плоская закраска с использованием алгоритма художника.
        :return: Список спроецированных на экран точек-вершин полигонов в виде ((A, B, C), цвет QColor).
        """
        data = []

        def key_func(polygon):
            """
            Среднее значение координаты Z.
            """
            return (polygon.a.z + polygon.b.z + polygon.c.z) / 3

        # Для отрисовки используется алгоритм художника. В его основе лежит идея,
        # отрисовки сначала наиболее удаленных объектов. Удаленность определяется
        # по среднему значению координаты Z.
        self.polygons.sort(key=key_func)

        for polygon in self.polygons:
            pol_coord = polygon.convertion_two_dim()
            try:
                pol_color = self.flat_shading_color(polygon)
            except ZeroDivisionError:
                # Чтобы не падало))
                pol_color = None

            data.append((pol_coord, pol_color))

        return data

    def flat_shading_color(self, polygon):
        """
        Плоская закраска.
        Вычисление цвета и его яркости. При одной и той же итенсивности света полигон освещён максимально ярко,
        если свет ему перпендикулярен. Если косинус угла между вектором к источнику и нормалью поверхности
        неотрицателен, то данная поверхность внешняя, в противном случае внутренняя
        :param polygon:
        :return:
        """

        def vec_length(vec):
            return math.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])

        # front_color = QColor(255, 0, 0)
        # inner_color = QColor(0, 0, 255)

        # Вектор к наблюдателю (имитация источника света).
        # (В развернутой систем координат наблюдатель находится на оси z).
        light_vec = (0, 0, 1)
        # light_vec = (self.light_source.x, self.light_source.y, self.light_source.z)

        norm = self.surface_normal(polygon)

        # cos = (obs[0]*norm[0] + obs[1]*norm[1] + obs[2]*norm[2]) / (vec_length(norm) * vec_length(obs))
        def get_cos(v1, v2):
            return (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]) / (vec_length(v1) * vec_length(v2))

        cos = get_cos(light_vec, norm)

        if cos < 0:
            # Для внутренней поверхности.
            color = self.inner_color
            cos = -cos
        else:
            # Для внешней.
            color = self.front_color

        polygon_color = QColor(color.red() * cos,
                               color.green() * cos,
                               color.blue() * cos)

        return polygon_color

    def flat_shading_z_buffer(self):
        # Создание Z-буфера (массив размером ширина х высота)
        z_buffer = []
        for i in range(self.width):
            z_buffer.append([])
            for j in range(self.height):
                z_buffer.append(None)

    def gouraud_shading(self):
        """
        Закраска по Гуро.
        :return:
        """
        """
        Вычисление нормали в вершине. Производится обход всех полигонов. Для каждого
        полигона вычисляется вектор нормали, затем добавляется во все вершины, входящие
        в полигон. После этого для каждой вершины вычисляется усредненная нормаль. При
        этом производится однократный обход всех полигонов и всех вершин.
        """
        v_map = {}  # Словарь "вершина : список нормалей смежных полигонов".

        # Просматриваем каждый полигон и вычисляем его нормаль.
        for pol in self.polygons:
            norm = self.surface_normal(pol)
            # Для каждого полигона рассматриваем его вершины и записываем нормали.
            for v in pol.vertices:
                if v in v_map:
                    v_map[v].append(norm)
                else:
                    v_map[v] = []
                    v_map[v].append(norm)

        v_normals = {}  # Словарь для хранения усредненной нормали вершины.

        for key in v_map.keys():
            v_normals[key] = self.average_vector(v_map[key])

    def average_vector(self, vectors):
        pass

    def phong_shading(self):
        pass