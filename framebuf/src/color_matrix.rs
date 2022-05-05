#[rustfmt::skip]
impl Primaries {
    pub(crate) fn to_xyz(self, white: Whitepoint) -> RowMatrix {
        use Primaries::*;
        // Rec.BT.601
        // https://en.wikipedia.org/wiki/Color_spaces_with_RGB_primaries#Specifications_with_RGB_primaries
        let xy: [[f32; 2]; 3] = match self {
            Bt601_525 | Smpte240 => [[0.63, 0.34], [0.31, 0.595], [0.155, 0.07]],
            Bt601_625 => [[0.64, 0.33], [0.29, 0.6], [0.15, 0.06]],
            Bt709 => [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]],
            Bt2020 | Bt2100 => [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
        };

        // A column of CIE XYZ intensities for that primary.
        let xyz = |[x, y]: [f32; 2]| {
            [x / y, 1.0, (1.0 - x - y)/y]
        };

        let xyz_r = xyz(xy[0]);
        let xyz_g = xyz(xy[1]);
        let xyz_b = xyz(xy[2]);

        // Virtually, N = [xyz_r | xyz_g | xyz_b]
        // As the unweighted conversion matrix for:
        //  XYZ = N · RGB
        let RowMatrix(n1) = ColMatrix([xyz_r, xyz_g, xyz_b]).inv();

        // http://www.brucelindbloom.com/index.html
        let w = white.to_xyz();

        // s is the weights that give the whitepoint when converted to xyz.
        // That is we're solving:
        //  W = N · S
        let s = [
            (w[0]*n1[0] + w[1]*n1[1] + w[2]*n1[2]),
            (w[0]*n1[3] + w[1]*n1[4] + w[2]*n1[5]),
            (w[0]*n1[6] + w[1]*n1[7] + w[2]*n1[8]),
        ];

        RowMatrix([
            s[0]*xyz_r[0], s[1]*xyz_g[0], s[2]*xyz_b[0],
            s[0]*xyz_r[1], s[1]*xyz_g[1], s[2]*xyz_b[1],
            s[0]*xyz_r[2], s[1]*xyz_g[2], s[2]*xyz_b[2],
        ])
    }
}

/// A column major matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ColMatrix([[f32; 3]; 3]);

/// A row major matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RowMatrix([f32; 9]);

#[rustfmt::skip]
impl ColMatrix {
    fn adj(self) -> RowMatrix {
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

    fn det(self) -> f32 {
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

    pub(crate) fn new(rows: [f32; 9]) -> RowMatrix {
        RowMatrix(rows)
    }

    pub(crate) fn diag(x: f32, y: f32, z: f32) -> Self {
        RowMatrix([
            x, 0., 0.,
            0., y, 0.,
            0., 0., z,
        ])
    }

    #[allow(clippy::many_single_char_names)]
    pub(crate) fn transpose(self) -> Self {
        let [a, b, c, d, e, f, g, h, i] = self.into_inner();

        RowMatrix([
            a, d, g,
            b, e, h,
            c, f, i,
        ])
    }

    pub(crate) fn inv(self) -> RowMatrix {
        ColMatrix::from(self).inv()
    }

    pub(crate) fn det(self) -> f32 {
        ColMatrix::from(self).det()
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

    pub(crate) fn into_inner(self) -> [f32; 9] {
        self.0
    }

    #[rustfmt::skip]
    pub(crate) fn into_mat3x3_std140(self) -> [f32; 12] {
        // std140, always pad components to 16 bytes.
        // matrix is an array of its columns.
        let matrix = self.into_inner();

        [
            matrix[0], matrix[3], matrix[6], 0.0,
            matrix[1], matrix[4], matrix[7], 0.0,
            matrix[2], matrix[5], matrix[8], 0.0,
        ]
    }
}

#[rustfmt::skip]
impl From<ColMatrix> for RowMatrix {
    fn from(ColMatrix(m): ColMatrix) -> RowMatrix {
        RowMatrix([
            m[0][0], m[1][0], m[2][0],
            m[0][1], m[1][1], m[2][1],
            m[0][2], m[1][2], m[2][2],
        ])
    }
}

#[rustfmt::skip]
impl From<RowMatrix> for ColMatrix {
    fn from(RowMatrix(r): RowMatrix) -> ColMatrix {
        ColMatrix([
            [r[0], r[3], r[6]],
            [r[1], r[4], r[7]],
            [r[2], r[5], r[8]],
        ])
    }
}

#[test]
fn matrix_ops() {
    let mat = RowMatrix::with_outer_product([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]);

    assert_eq!(mat, mat.transpose().transpose());
}
