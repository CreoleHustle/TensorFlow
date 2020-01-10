"""Tests for division with division imported from __future__.

This file should be exactly the same as division_past_test.py except
for the __future__ division line.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform

import numpy as np
import tensorflow as tf


class DivisionTestCase(tf.test.TestCase):

  def testDivision(self):
    """Test all the different ways to divide."""
    values = [1, 2, 7, 11]
    functions = (lambda x: x), tf.constant
    # TODO(irving): Test int8, int16 once we support casts for those.
    dtypes = np.int32, np.int64, np.float32, np.float64

    def check(x, y):
      if isinstance(x, tf.Tensor):
        x = x.eval()
      if isinstance(y, tf.Tensor):
        y = y.eval()
      self.assertEqual(x.dtype, y.dtype)
      self.assertEqual(x, y)
    with self.test_session():
      for dtype in dtypes:
        for x in map(dtype, values):
          for y in map(dtype, values):
            for fx in functions:
              for fy in functions:
                tf_x = fx(x)
                tf_y = fy(y)
                div = x / y
                tf_div = tf_x / tf_y
                check(div, tf_div)
                floordiv = x // y
                tf_floordiv = tf_x // tf_y
                check(floordiv, tf_floordiv)


if __name__ == "__main__":
  tf.test.main()
