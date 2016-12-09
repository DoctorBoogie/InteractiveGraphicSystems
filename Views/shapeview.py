from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath, QPolygonF
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QPoint


class ShapeView(QWidget):
    def __init__(self, parent=None):
        super(ShapeView, self).__init__(parent)
        self.polygons = []
        self.buffer = {}

    def paintEvent(self, event):
        pass

    def draw_polygons(self, event):
        if self.polygons:
            painter = QPainter(self)

            # painter.drawEllipse(0, 0, self.width()/2, self.width()/2)
            for (pa, pb, pc), color in self.polygons:
                a = QPoint(pa.x, pa.y)
                b = QPoint(pb.x, pb.y)
                c = QPoint(pc.x, pc.y)

                pen = QPen()

                if color:
                    pen.setColor(color)
                else:
                    pen.setColor(QColor(0, 0, 0))

                painter.setPen(pen)

                polygon = QPolygonF([a, b, c])

                painter.drawPolygon(polygon)

                if color:
                    path = QPainterPath()
                    path.addPolygon(polygon)

                    # painter.setBrush(QBrush(color))
                    painter.fillPath(path, QBrush(color))
                # print(pa, pb, pc)

    def draw_buffer(self, event):
        painter = QPainter(self)
        pen = QPen()

        for x, y, color in self.buffer.values():
            pen.setColor(color)

            painter.drawPoint(QPoint(x, y))


    def set_buffer(self, buffer):
        self.buffer = buffer
        self.paintEvent = self.draw_buffer

    def set_polygons(self, polygons):
        self.polygons = polygons
        self.paintEvent = self.draw_polygons