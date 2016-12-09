from PyQt5.QtWidgets import QColorDialog
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt

from Utility.shell import Shell
from Utility.torus import Torus

from Utility.structures import HomogeneousPoint


class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view

        init_front_color = QColor(Qt.red)
        init_inner_color = QColor(Qt.green)

        self.view.pushButton_front.setStyleSheet("background-color: %s" % init_front_color.name())
        self.view.pushButton_inner.setStyleSheet("background-color: %s" % init_inner_color.name())

        self.model.set_front_color(init_front_color)
        self.model.set_inner_color(init_inner_color)

        self.drawing_mode = self.model.wireframe_model

        self.update_all()

        # self.model.polygon_approximation()
        # self.view.shape_view.set_polygons(self.model.points_plane())

        self.connect_signals()

        self.set_limits(**self.model.shape.get_limits())

        self.view.show()

        modes = ['Каркас', 'Плоская закраска', 'Плоская закраска (Z-буфер)', 'Закраска Гуро']
        self.view.comboBox_mode.addItems(modes)

        shapes = ['Тор', 'Ракушка']
        self.view.comboBox_shape.addItems(shapes)

    def connect_signals(self):
        self.view.slider_observer_x.valueChanged.connect(self.on_observer_changed)
        self.view.slider_observer_y.valueChanged.connect(self.on_observer_changed)
        self.view.slider_observer_z.valueChanged.connect(self.on_observer_changed)

        self.view.slider_u_step.valueChanged.connect(self.on_steps_changed)
        self.view.slider_v_step.valueChanged.connect(self.on_steps_changed)

        self.view.slider_a_param.valueChanged.connect(self.on_params_changed)
        self.view.slider_b_param.valueChanged.connect(self.on_params_changed)

        self.view.slider_u.valueChanged.connect(self.on_params_changed)
        self.view.slider_v.valueChanged.connect(self.on_params_changed)

        self.view.pushButton_front.clicked.connect(self.on_color_front_changed)
        self.view.pushButton_inner.clicked.connect(self.on_color_inner_changed)

        self.view.comboBox_mode.currentIndexChanged.connect(self.on_mode_select)
        self.view.comboBox_shape.currentIndexChanged.connect(self.on_shape_select)

    def on_shape_select(self):
        index = self.view.comboBox_shape.currentIndex()
        if index == 0:
            self.model.set_shape(Torus())
        elif index == 1:
            self.model.set_shape(Shell())
        else:
            pass

        self.set_limits(**self.model.shape.get_limits())
        self.set_params(**self.model.shape.get_params())

        self.on_params_changed()
        self.on_steps_changed()

        self.update_all()

    def update_labels(self):
        self.view.label_a.setText("Параметр А: " + str(self.view.slider_a_param.value()))
        self.view.label_b.setText("Параметр B: " + str(self.view.slider_b_param.value()))

        self.view.label_u.setText("Параметр U: " + str(self.view.slider_u.value()))
        self.view.label_v.setText("Параметр V: " + str(self.view.slider_v.value()))

        self.view.label_u_step.setText("Шаг U: " + str(self.view.slider_u_step.value()))
        self.view.label_v_step.setText("Шаг V: " + str(self.view.slider_v_step.value()))

        self.view.label_x.setText("Координата X: " + str(self.view.slider_observer_x.value()))
        self.view.label_y.setText("Координата Y: " + str(self.view.slider_observer_y.value()))
        self.view.label_z.setText("Координата Z: " + str(self.view.slider_observer_z.value()))

    def on_mode_select(self):
        mode = self.view.comboBox_mode.currentIndex()
        if mode == 0:
            self.drawing_mode = self.model.wireframe_model
        elif mode == 1:
            self.drawing_mode = self.model.flat_shading
        elif mode == 2:
            self.drawing_mode = self.model.flat_shading_z_buffer
        elif mode == 3:
            self.drawing_mode = self.model.gouraud_shading
        else:
            pass

        self.update_all()

    def on_color_front_changed(self):
        col = QColorDialog.getColor()

        if col.isValid():
            self.view.pushButton_front.setStyleSheet("background-color: %s" % col.name())
            self.model.set_front_color(col)

            self.update_all()

    def on_color_inner_changed(self):
        col = QColorDialog.getColor()

        if col.isValid():
            self.view.pushButton_inner.setStyleSheet("background-color: %s" % col.name())
            self.model.set_inner_color(col)

            self.update_all()

    def on_observer_changed(self):
        self.update_labels()

        self.model.set_observer(HomogeneousPoint(self.view.slider_observer_x.value(),
                                                 self.view.slider_observer_y.value(),
                                                 self.view.slider_observer_z.value()))
        self.update_all()

    def on_steps_changed(self):
        self.update_labels()

        self.model.set_steps(self.view.slider_u_step.value(), self.view.slider_v_step.value())
        self.update_all()

    def on_params_changed(self):
        self.update_labels()

        self.model.shape.set_params(self.view.slider_a_param.value(),
                                    self.view.slider_b_param.value(),
                                    self.view.slider_u.value(),
                                    self.view.slider_v.value())
        self.update_all()

    def update_all(self):
        self.model.polygon_approximation()
        if self.view.comboBox_mode.currentIndex() in [0, 1]:
            self.view.shape_view.set_polygons(self.drawing_mode())
        elif self.view.comboBox_mode.currentIndex() in [2, 3]:
            self.view.shape_view.set_buffer(self.drawing_mode())
        self.view.shape_view.update()

    def set_limits(self, u_min, u_max, v_min, v_max):
        self.view.slider_u.setMaximum(u_max)
        self.view.slider_u.setMinimum(u_min)
        self.view.slider_v.setMaximum(v_max)
        self.view.slider_v.setMinimum(v_min)

        self.view.label_u_min.setText(str(u_min))
        self.view.label_u_max.setText(str(u_max))
        self.view.label_v_min.setText(str(v_min))
        self.view.label_v_max.setText(str(v_max))

    def set_params(self, u, v, a, b):
        self.view.slider_u.setValue(u)
        self.view.slider_v.setValue(v)
        self.view.slider_a_param.setValue(a)
        self.view.slider_b_param.setValue(b)