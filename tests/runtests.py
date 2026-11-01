# -*- coding: utf-8 -*-

import numpy as np
import os.path
import sys
import unittest

SRCDIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(SRCDIR)

import astrofocus as focus
from astrofocus import *

class TestAstroFocus(unittest.TestCase):

    def test_image_roi(self):
        for kwds in [{'xcen':56.1, 'ycen':71.3, 'width':-1, 'height':11.0},
                     {'xcen':56.1, 'ycen':71.3, 'width':2.7, 'height':-2.0},
                     {'xcen':56.1, 'ycen':71.3, 'width':3, 'height':11.0},
                     {'xcen':56.1, 'ycen':71.6, 'width':6, 'height':8.0}]:
            if 'xmin' in kwds:
                xmin = round(kwds['xmin'])
            elif 'xmax' in kwds and 'width' in kwds:
                xmin = round(kwds['xmax']) - round(kwds['width']) + 1
            elif 'xmax' in kwds and 'xcen' in kwds:
                xmin = round(2*kwds['xcen']) - round(kwds['xmax'])
            elif 'width' in kwds and 'xcen' in kwds:
                xmin = round(kwds['xcen'] - (round(kwds['width']) - 1)/2)
            else:
                xmin = None
            if 'xmax' in kwds:
                xmax = round(kwds['xmax'])
            elif 'xmin' in kwds and 'width' in kwds:
                xmax = round(kwds['xmin']) + round(kwds['width']) - 1
            elif 'xmin' in kwds and 'xcen' in kwds:
                xmax = round(2*kwds['xcen']) - round(kwds['xmin'])
            elif 'width' in kwds and 'xcen' in kwds:
                width = round(kwds['width'])
                xmax = round(kwds['xcen'] - (width - 1)/2) + width - 1
            else:
                xmax = None
            if 'ymin' in kwds:
                ymin = round(kwds['ymin'])
            elif 'ymax' in kwds and 'height' in kwds:
                ymin = round(kwds['ymax']) - round(kwds['height']) + 1
            elif 'ymax' in kwds and 'ycen' in kwds:
                ymin = round(2*kwds['ycen']) - round(kwds['ymax'])
            elif 'height' in kwds and 'ycen' in kwds:
                ymin = round(kwds['ycen'] - (round(kwds['height']) - 1)/2)
            else:
                ymin = None
            if 'ymax' in kwds:
                ymax = round(kwds['ymax'])
            elif 'ymin' in kwds and 'height' in kwds:
                ymax = round(kwds['ymin']) + round(kwds['height']) - 1
            elif 'ymin' in kwds and 'ycen' in kwds:
                ymax = round(2*kwds['ycen']) - round(kwds['ymin'])
            elif 'height' in kwds and 'ycen' in kwds:
                height = round(kwds['height'])
                ymax = round(kwds['ycen'] - (height - 1)/2) + height - 1
            else:
                ymax = None

            roi = ImageROI(**kwds)

            self.assertEqual(type(roi.xmin()),   int)
            self.assertEqual(type(roi.xmax()),   int)
            self.assertEqual(type(roi.xcen()),   float)
            self.assertEqual(type(roi.width()),  int)
            self.assertEqual(type(roi.xrange()), range)

            self.assertEqual(type(roi.ymin()),   int)
            self.assertEqual(type(roi.ymax()),   int)
            self.assertEqual(type(roi.ycen()),   float)
            self.assertEqual(type(roi.height()), int)
            self.assertEqual(type(roi.yrange()), range)

            if xmin > xmax or ymin > ymax:
                self.assertTrue(roi.is_empty())
            else:
                self.assertFalse(roi.is_empty())
            if xmin > xmax:
                self.assertEqual(roi.xmin(), 0)
                self.assertEqual(roi.xmax(), -1)
                self.assertEqual(roi.width(), 0)
                self.assertEqual(roi.xrange(), range(0, 0))
            else:
                self.assertEqual(roi.xmin(), xmin)
                self.assertEqual(roi.xmax(), xmax)
                self.assertEqual(roi.xcen(), (xmin + xmax)/2)
                self.assertEqual(roi.width(), xmax - xmin + 1)
                self.assertEqual(roi.xrange(), range(xmin, xmax + 1))
            if ymin > ymax:
                self.assertEqual(roi.ymin(), 0)
                self.assertEqual(roi.ymax(), -1)
                self.assertEqual(roi.height(), 0)
                self.assertEqual(roi.yrange(), range(0, 0))
            else:
                self.assertEqual(roi.ymin(), ymin)
                self.assertEqual(roi.ymax(), ymax)
                self.assertEqual(roi.ycen(), (ymin + ymax)/2)
                self.assertEqual(roi.height(), ymax - ymin + 1)
                self.assertEqual(roi.yrange(), range(ymin, ymax + 1))

            if xmin <= xmax and ymin <= ymax:
                # ROI is not empty.
                dx, dy = 2, -3
                roi.move_by(dx, dy)
                self.assertEqual(roi.xmin(), xmin + dx)
                self.assertEqual(roi.xmax(), xmax + dx)
                self.assertEqual(roi.ymin(), ymin + dy)
                self.assertEqual(roi.ymax(), ymax + dy)
                roi.move_by(-dx, -dy)
                self.assertEqual(roi.xmin(), xmin)
                self.assertEqual(roi.xmax(), xmax)
                self.assertEqual(roi.ymin(), ymin)
                self.assertEqual(roi.ymax(), ymax)

                self.assertFalse(roi.is_empty())

    #def test_split(self):
    #    s = 'hello world'
    #    self.assertEqual(s.split(), ['hello', 'world'])
    #    # check that s.split fails when the separator is not a string
    #    with self.assertRaises(TypeError):
    #        s.split(2)

if __name__ == '__main__':
    unittest.main()
