#![allow(dead_code)] // Mostly imported code from another project. Bare minimum linear algebra.

/// A column major matrix.
///
/// FIXME: const everything, in particular matrix inversion, but this is blocked on floating point
/// arithmetic in constants.

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ColMatrix(pub(crate) [[f32; 3]; 3]);

/// A row major matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RowMatrix(pub(crate) [f32; 9]);

#[rustfmt::skip]
impl ColMatrix {
    pub(crate) fn adj(self) -> RowMatrix {
        let m = self.0;

        let det = |c1: usize, c2: usize, r1: usize, r2: usize| {
            m[c1][r1] * m[c2][r2] - m[c2][r1] * m[c1][r2]
        };

        RowMatrix([
            det(1, 2, 1, 2), -det(1, 2, 0, 2), det(1, 2, 0, 1),
            -det(0, 2, 1, 2), det(0, 2, 0, 2), -det(0, 2, 0, 1),
            det(0, 1, 1, 2), -det(0, 1, 0, 2), det(0, 1, 0, 1),
        ])
    }

    pub(crate) fn det(self) -> f32 {
        let det2 = |ma, mb, na, nb| {
            ma * nb - na * mb
        };
        let [x, y, z] = self.0;
        x[0] * det2(y[1], y[2], z[1], z[2])
            - x[1] * det2(y[0], y[2], z[0], z[2])
            + x[2] * det2(y[0], y[1], z[0], z[1])
    }

    pub(crate) fn inv(self) -> RowMatrix {
        let RowMatrix(adj) = self.adj();
        let det_n = self.det();

        RowMatrix([
            adj[0] / det_n, adj[1] / det_n, adj[2] / det_n,
            adj[3] / det_n, adj[4] / det_n, adj[5] / det_n,
            adj[6] / det_n, adj[7] / det_n, adj[8] / det_n,
        ])
    }

    #[rustfmt::skip]
    pub(crate) const fn to_row(self) -> RowMatrix {
        let ColMatrix(m) = self;

        RowMatrix([
            m[0][0], m[1][0], m[2][0],
            m[0][1], m[1][1], m[2][1],
            m[0][2], m[1][2], m[2][2],
        ])
    }

    pub(crate) fn mul_vec(&self, vec: [f32; 3]) -> [f32; 3] {
        let ColMatrix(m) = self;
        let [a, b, c] = vec;

        [
            a*m[0][0] + b*m[1][0] + c*m[2][0],
            a*m[0][1] + b*m[1][1] + c*m[2][1],
            a*m[0][2] + b*m[1][2] + c*m[2][2],
        ]
    }
}

#[rustfmt::skip]
impl RowMatrix {
    pub(crate) fn with_outer_product(a: [f32; 3], b: [f32; 3]) -> Self {
        RowMatrix([
            a[0]*b[0], a[0]*b[1], a[0]*b[2],
            a[1]*b[0], a[1]*b[1], a[1]*b[2],
            a[2]*b[0], a[2]*b[1], a[2]*b[2],
        ])
    }

    pub(crate) const fn new(rows: [f32; 9]) -> RowMatrix {
        RowMatrix(rows)
    }

    pub(crate) const fn diag(x: f32, y: f32, z: f32) -> Self {
        RowMatrix([
            x, 0., 0.,
            0., y, 0.,
            0., 0., z,
        ])
    }

    #[allow(clippy::many_single_char_names)]
    pub(crate) const fn transpose(self) -> Self {
        let [a, b, c, d, e, f, g, h, i] = self.into_inner();

        RowMatrix([
            a, d, g,
            b, e, h,
            c, f, i,
        ])
    }

    pub(crate) fn inv(self) -> RowMatrix {
        self.to_col().inv()
    }

    pub(crate) fn det(self) -> f32 {
        self.to_col().det()
    }

    /// Multiply with a homogeneous point.
    /// Note: might produce NaN if the matrix isn't a valid transform and may produce infinite
    /// points.
    pub(crate) fn multiply_point(self, point: [f32; 2]) -> [f32; 2] {
        let [x, y, z] = self.multiply_column([point[0], point[1], 1.0]);
        [x / z, y / z]
    }

    /// Calculate self · other
    pub(crate) fn multiply_column(self, col: [f32; 3]) -> [f32; 3] {
        let x = &self.0[0..3];
        let y = &self.0[3..6];
        let z = &self.0[6..9];

        let dot = |r: &[f32], c: [f32; 3]| {
            r[0] * c[0] + r[1] * c[1] + r[2] * c[2]
        };

        [dot(x, col), dot(y, col), dot(z, col)]
    }

    /// Calculate self · other
    pub(crate) fn multiply_right(self, ColMatrix([a, b, c]): ColMatrix) -> ColMatrix {
        ColMatrix([
            self.multiply_column(a),
            self.multiply_column(b),
            self.multiply_column(c),
        ])
    }

    pub(crate) const fn into_inner(self) -> [f32; 9] {
        self.0
    }

    #[rustfmt::skip]
    pub(crate) const fn into_mat3x3_std140(self) -> [f32; 12] {
        // std140, always pad components to 16 bytes.
        // matrix is an array of its columns.
        let matrix = self.into_inner();

        [
            matrix[0], matrix[3], matrix[6], 0.0,
            matrix[1], matrix[4], matrix[7], 0.0,
            matrix[2], matrix[5], matrix[8], 0.0,
        ]
    }

    #[rustfmt::skip]
    pub(crate) const fn to_col(self) -> ColMatrix {
        let RowMatrix(r) = self;

        ColMatrix([
            [r[0], r[3], r[6]],
            [r[1], r[4], r[7]],
            [r[2], r[5], r[8]],
        ])
    }

    pub(crate) fn mul_vec(&self, vec: [f32; 3]) -> [f32; 3] {
        self.multiply_column(vec)
    }
}

impl From<ColMatrix> for RowMatrix {
    fn from(col: ColMatrix) -> RowMatrix {
        col.to_row()
    }
}

impl From<RowMatrix> for ColMatrix {
    fn from(row: RowMatrix) -> ColMatrix {
        row.to_col()
    }
}

#[test]
fn matrix_ops() {
    let mat = RowMatrix::with_outer_product([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]);

    assert_eq!(mat, mat.transpose().transpose());
}
