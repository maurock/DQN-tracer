#include "pybind11/pybind11.h"
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <math.h>
#include "pybind11/stl.h"
#include <map>
#include <fstream>
namespace py = pybind11;

// Settings
int action_space = 72;
bool equally_sized_patches = false;
bool not_equally_sized_patches = true;
bool double_action_space = false; //based on not equally sized patches

struct Vec {
	float x, y, z;                  // position, also color (r,g,b)
	Vec(float x_ = 0, float y_ = 0, float z_ = 0) {
		x = x_;
		y = y_;
		z = z_;
	}
	Vec operator+(Vec b) const {
		return Vec(x + b.x, y + b.y, z + b.z);
	}
	Vec operator-(Vec b) const {
		return Vec(x - b.x, y - b.y, z - b.z);
	}
	Vec operator*(float b) const {
		return Vec(x * b, y * b, z * b);
	}
	Vec operator*(double b) const {
		return Vec(x * b, y * b, z * b);
	}
	Vec operator*(int b) const {
		return Vec(x * b, y * b, z * b);
	}
	Vec mult(Vec b) {
		return Vec(x * b.x, y * b.y, z * b.z);
	}
	Vec norm() {
		return *this = *this * (1 / sqrt(x * x + y * y + z * z));
	}
	float dot(Vec b) {
		return x * b.x + y * b.y + z * b.z;
	}
	Vec cross(Vec b) {
		return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);
	}
	/*Vec operator%(Vec &b) {
		return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);
	} // cross
	Vec operator%(const Vec &b) {
		return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);
	} */// cross
	float magnitude() {
		return sqrt(x * x + y * y + z * z);
	}
	float get_max() {
		return x > y && x > z ? x : y > z ? y : z;
	}
	float get_x() {
		return x;
	}
	float get_y() {
		return y;
	}
	float get_z() {
		return z;
	}

};

using Action = int;
using Direction = Vec;

struct Ray {
	Vec o, d;
	Ray(Vec o_, Vec d_) : o(o_), d(d_) {}
};


class Hitable {			// a pure virtual function makes sure we always override the function hit
public:
	virtual float intersect(Ray r) = 0;
	virtual Vec normal(Ray r, Vec x) = 0;
	virtual Vec get_e() = 0;
	virtual Vec get_c() = 0;

};

class PyHitable : public Hitable {
public:
	using Hitable::Hitable;

	// Trampoline
	float intersect(Ray r) override {
		PYBIND11_OVERLOAD_PURE(
			float,   // return type
			Hitable,
			intersect,
			r
		);
	}

	Vec normal(Ray r, Vec x) override {
		PYBIND11_OVERLOAD_PURE(
			Vec,   // return type
			Hitable,
			normal,
			r, x
		);
	}

	Vec get_e() override {
		PYBIND11_OVERLOAD_PURE(
			Vec,   // return type
			Hitable,
			get_e
		);
	}


	Vec get_c() override {
		PYBIND11_OVERLOAD_PURE(
			Vec,   // return type
			Hitable,
			get_c
		);
	}
};

class Rectangle_xz : public Hitable {
public:
	float x1, x2, z1, z2, y;
	Vec e, c;         // emission, color
	Rectangle_xz(float x1_, float x2_, float z1_, float z2_, float y_,
		Vec e_, Vec c_) :
		x1(x1_), x2(x2_), z1(z1_), z2(z2_), y(y_), e(e_), c(c_) {}

	float intersect(Ray r) { // returns distance, 0 if no hit
		float t = (y - r.o.y) / r.d.y;		// ray.y = t* dir.y
		float x = r.o.x + r.d.x * t;
		float z = r.o.z + r.d.z * t;
		if (x < x1 || x > x2 || z < z1 || z > z2 || t < 0) {
			t = 0;
			return 0;
		}
		else {
			return t;
		}
	}

	Vec normal(Ray r, Vec x) {
		Vec n = Vec(0, 1, 0);
		return n.dot(r.d) < 0 ? n : n * -1;
	}

	Vec get_c() {
		return c;
	}

	Vec get_e() {
		return e;
	}
};

class Rectangle_xy : public Hitable {
public:
	float x1, x2, y1, y2, z;
	Vec e, c;         // emission, color
	Rectangle_xy(float x1_, float x2_, float y1_, float y2_, float z_, Vec e_, Vec c_) :
		x1(x1_), x2(x2_), y1(y1_), y2(y2_), z(z_), e(e_), c(c_) {}

	float intersect(Ray r) { // returns distance, 0 if no hit
		float t = (z - r.o.z) / r.d.z;
		float x = r.o.x + r.d.x * t;
		float y = r.o.y + r.d.y * t;
		if (x < x1 || x > x2 || y < y1 || y > y2 || t < 0) {
			t = 0;
			return 0;
		}
		else {
			return t;
		}
	}

	Vec normal(Ray r, Vec x) {
		Vec n = Vec(0, 0, 1);
		return n.dot(r.d) < 0 ? n : n * -1;
	}

	Vec get_c() {
		return c;
	}

	Vec get_e() {
		return e;
	}
};

class Rectangle_yz : public Hitable {
public:
	float y1, y2, z1, z2, x;
	Vec e, c;         // emission, color
	Rectangle_yz(float y1_, float y2_, float z1_, float z2_, float x_, Vec e_, Vec c_) :
		y1(y1_), y2(y2_), z1(z1_), z2(z2_), x(x_), e(e_), c(c_) {}

	float intersect(Ray r) { // returns distance, 0 if no hit
		float t = (x - r.o.x) / r.d.x;
		float y = r.o.y + r.d.y * t;
		float z = r.o.z + r.d.z * t;
		if (y < y1 || y > y2 || z < z1 || z > z2 || t < 0) {
			t = 0;
			return 0;
		}
		else {
			return t;
		}
	}

	Vec normal(Ray r, Vec x) {
		Vec n = Vec(1, 0, 0);
		return n.dot(r.d) < 0 ? n : n * -1;
	}

	Vec get_c() {
		return c;
	}

	Vec get_e() {
		return e;
	}
};

class Rectangle_tilted : public Hitable {
public:
	Vec p1, p2, p3;
	Vec e, c;         // emission, color
	Rectangle_tilted(Vec p1_, Vec p2_, Vec p3_, Vec e_, Vec c_) :
		p1(p1_), p2(p2_), p3(p3_), e(e_), c(c_) {}

	float intersect(Ray r) 	{
		Vec n = (Vec(p1 - p2).cross(Vec(p1 - p3))).norm();
		// assuming vectors are all normalized
		if ((r.d).dot(n) != 0) {
			float  t = ((p1 - r.o).dot(n)) / ((r.d).dot(n));
			float x = r.o.x + r.d.x * t;
			float y = r.o.y + r.d.y * t;
			float z = r.o.z + r.d.z * t;

			if (x < p2.x && x > p1.x && z < p2.z && z > p1.z && y < p3.y && y > p1.y && t > 0) {
				return t;
			}
		}
		return 0;
	}

	Vec normal(Ray r, Vec x) {
		Vec n = (Vec(p1 - p2).cross(Vec(p1 - p3))).norm();
		return n.dot(r.d) < 0 ? n : n * -1;
	}

	Vec get_c() {
		return c;
	}

	Vec get_e() {
		return e;
	}
};

Hitable *scene[19] = {
	new Rectangle_xy(1, 99, 0, 81.6, -25, Vec(),Vec(.75, .75, .75)), 				// Front2
	new Rectangle_xy(1, 99, 0, 81.6, 0, Vec(),Vec(.75, .75, .75)), 					// Front
	new Rectangle_xy(1, 99, 0, 81.6, 170, Vec(), Vec(.75, .75, .75)),				// Back
	new Rectangle_yz(0, 81.6, 0, 170, 1, Vec(), Vec(.25, .75, .25)),				// Left, green
	new Rectangle_yz(0, 81.6, 0, 170, 99, Vec(), Vec(.75, .25, .25)),				// Right, red
	new Rectangle_xz(1, 99, 0, 170, 0, Vec(), Vec(.75, .75, .75)),					// Bottom
	new Rectangle_xz(1, 99, 0, 170, 81.6, Vec(), Vec(.75, .75, .75)),				// Top
	//new Rectangle_xz(32, 68, 63, 96, 81.595, Vec(12, 12, 12), Vec()),				// Light 1
	//new Rectangle_xz(88, 99, 63, 88, 0.01, Vec(12, 12, 12), Vec()),				// Light 2
	new Rectangle_yz(0, 60, 62, 90, 1.01, Vec(12,12,12), Vec()),					// Light 3

	new Rectangle_xy(12, 42, 0, 50, 32, Vec(), Vec(.95,.95,.95)),					// Tall box
	new Rectangle_xy(12, 42, 0, 50, 62, Vec(), Vec(.95,.95,.95)),
	new Rectangle_yz(0, 50, 32, 62, 12, Vec(), Vec(.95,.95,.95)),
	new Rectangle_yz(0, 50, 32, 62, 42 , Vec(), Vec(.95,.95,.95)),
	new Rectangle_xz(12, 42, 32, 62, 50, Vec(), Vec(.95,.95,.95)),

	new Rectangle_xy(63, 88, 0, 25, 63, Vec(), Vec(.95,.95,.95)),					// Short box
	new Rectangle_xy(63, 88, 0, 25, 88, Vec(), Vec(.95,.95,.95)),
	new Rectangle_yz(0, 25, 63, 88, 63, Vec(), Vec(.95,.95,.95)),
	new Rectangle_yz(0, 25, 63, 88, 88, Vec(), Vec(.95,.95,.95)),
	new Rectangle_xz(63, 88, 63, 88, 25, Vec(), Vec(.95,.95,.95)),

	new Rectangle_tilted(Vec(1.1,0,60),Vec(10,0,100),Vec(1,60,60),Vec(), Vec(.75, .25, .25))		// Tilted plane
};

class Camera {
public:
	// lookfrom is the origin
	// lookat is the point to look at
	// vup, the view up vector to project on the new plane when we incline it. We can also tilt
	// the plane
	Camera(Vec lookfrom, Vec lookat, Vec vup, float vfov, float aspect) {// vfov is top to bottom in degrees, field of view on the vertical axis
		Vec w, u, v;
		float theta = vfov * 3.141592653 / 180;	// convert to radiants
		float half_height = tan(theta / 2);
		float half_width = aspect * half_height;
		origin = lookfrom;
		w = (lookat - lookfrom).norm();
		u = (w.cross(vup)).norm();
		v = (u.cross(w));

		lower_left_corner = origin - u * half_width - v * half_height + w;
		horizontal = u * (half_width * 2);
		vertical = v * (half_height * 2);
	}

	Ray get_ray(float s, float t) {
		return Ray(origin,
			lower_left_corner + horizontal * s + vertical * t - origin);
	}

	Vec origin;
	Vec lower_left_corner;
	Vec horizontal;
	Vec vertical;
};

Vec importanceSampling_scattering(Vec nl) {

	// COSINE-WEIGHTED SAMPLING
	float r1 = 2 * 3.141592653 *  rand() / float(RAND_MAX);		// get random angle
	float r2 = rand() / float(RAND_MAX);			// get random distance from center
	float r2s = sqrt(r2);
	// Create orthonormal coordinate frame for scattered ray
	Vec w = nl;			// w = normal
	Vec u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)).cross(w)).norm();
	Vec v = w.cross(u);
	return (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();
}

float clamp(float x) {
	return x < 0 ? 0 : x > 1 ? 1 : x;
}

struct Hit_record {
	int id;
	int old_id;
	float t;
	bool hit;
	Vec nl;
	Vec e;
	Vec c;
	float prob;
	float BRDF;
	float costheta;
	std::tuple<float, float, float, int, int, int> old_state;
	int old_action;

	int get_id() {
		return id;
	}
	int get_oldid() {
		return old_id;
	}
	float get_t() {
		return t;
	}
	bool get_hit() {
		return hit;
	}
	Vec get_nl() {
		return nl;
	}
	Vec get_e() {
		return e;
	}
	Vec get_c() {
		return c;
	}
	float get_prob() {
		return prob;
	}
	std::tuple<float, float, float, int, int, int> get_oldstate() {
		return old_state;
	}
	int get_oldaction() {
		return old_action;
	}
	float get_BRDF() {
		return BRDF;
	}
	float get_costheta() {
		return costheta;
	}
	void set_id(int id_) {
		id = id_;
	}
	void set_oldid(int oldid_) {
		old_id = oldid_;
	}
	void set_oldaction(int oldaction_) {
		old_action = oldaction_;
	}
	void set_t(float t_) {
		t = t_;
	}
	void set_hit(bool hit_) {
		hit = hit_;
	}
	void set_prob(float prob_) {
		prob = prob_;
	}
	void set_BRDF(float BRDF_) {
		BRDF = BRDF_;
	}
	void set_costheta(float costheta_) {
		costheta = costheta_;
	}
};

bool intersection(Ray &r, Hit_record& hit_rec, int N_OBJ) {
	float d;
	float inf = hit_rec.t = 1e20;
	for (int i = 0; i < N_OBJ; i++) {
		if ((d = scene[i]->intersect(r)) && d < hit_rec.t && i != hit_rec.old_id) {	// Distance of hit point
			hit_rec.t = d;
			hit_rec.id = i;
			hit_rec.hit = true;
		}
	}
	// Return the closest intersection, as a bool
	return hit_rec.t < inf;
}

Vec hittingPoint(Ray r, Hit_record& hit_rec, int N_OBJ) {
	if (!intersection(r, hit_rec, N_OBJ))
		return Vec();
	hit_rec.nl = scene[hit_rec.id]->normal(r, Vec(0, 0, 0));
	hit_rec.e = scene[hit_rec.id]->get_e();
	hit_rec.c = scene[hit_rec.id]->get_c();

	Vec x = (r.o + r.d * (hit_rec.t - 0.01));// ray intersection point (t calculated in intersect())
	return x;
}

Vec spherToCart(Vec& spher) {
	return Vec(sin(spher.y)*sin(spher.z), sin(spher.y)*cos(spher.z), cos(spher.y));
	//return Vec(sin(spher.y)*sin(spher.z), cos(spher.y), sin(spher.y)*cos(spher.z));
}

// convert Cartesian coordinates into spherical
Vec cartToSpher(Vec& cart) {
	//return Vec(1, atan((sqrt(cart.x*cart.x + cart.y*cart.y)) / cart.z), atan2(cart.x, cart.y));     // (radius, theta (vertical), phi (orizontal))
	//py::print("cart.x", cart.x); 
	//py::print("cart.y", cart.y);
	//py::print("cart.z", cart.z);
	//py::print("y, theta", atan2((sqrt(cart.x*cart.x + cart.z*cart.z)), cart.y));
	//py::print("z, phi", atan2(cart.x, cart.z)); 
	//return Vec(1, atan2((sqrt(cart.x*cart.x + cart.z*cart.z)) , cart.y), atan2(cart.x,cart.z));     // (radius, theta (vertical), phi (orizontal))
	return Vec(1, atan((sqrt(cart.x*cart.x + cart.y*cart.y)) / cart.z), atan2(cart.x, cart.y));
}

std::map<Action, Direction> initialize_dictAction(std::string s) {
	std::map<Action, Direction> dict;
	//std::map<Action, Direction> &addrDict = *dictAction;
	std::ifstream myFile;
	myFile.open(s);
	std::string x, y, z;
	int count = 0;
	while (myFile.good()) {
		if (std::getline(myFile, x, ',') && std::getline(myFile, y, ',') && std::getline(myFile, z, '\n')) {
			dict[count] = Vec(std::stod(x), std::stod(y), std::stod(z));
			count += 1;
		}
	}
	myFile.close();
	return dict;
}

Vec getTangent(Vec& normal) {
	Vec new_vector = Vec(normal.x + 1, normal.y + 1, normal.z + 1);		// This cannot be zero, or parallel to the normal.
															// Usually I need to check, but since all my tangents are (0,0,1),
															// (0,1,0) and (1,0,0), I don't need to check in this SPECIFIC CASE.
	return (normal.cross(new_vector.norm()));
}

// Get the patch height for double action space, action space = 72
float getHeight(int action) {
	if (action >= 60) {
		return 0.408248;
	}
	else if (action >= 48 && action < 60) {
		return 0.16915;
	}
	else if (action >= 36 && action < 48) {
		return 0.12975;
	}
	else if (action >= 24 && action < 36) {
		return 0.1094;
	}
	else if (action >= 12 && action < 24) {
		return 0.0964;
	}
	else if (action < 12) {
		return 0.08713;
	}
}

// Get row and column index from action index
int* getRowColumn(int subaction) {
	static int x[2];
	x[0] = floor((float)subaction / 4);
	x[1] = subaction % 4;
	return x;
}

Vec DQNScattering(std::map<Action, Direction> *dictAction, Vec &nl, int& action, int double_action) {
	std::map<Action, Direction> &addrDictAction = *dictAction;
	
	// Create temporary coordinates system
	Vec w = nl.norm();
	const Vec& u = getTangent(w).norm();
	const Vec& v = (w.cross(u)).norm();

	Vec point_old_coord;

	// hitobj.prob = prob;	// 1/ ( (q/tot) * (1/ (2 * pi) / 24))
	point_old_coord = addrDictAction[action];

	// Scatter random inside the selected patch, convert to spherical coordinates for semplicity and then back to cartesian
	Vec spher_coord = cartToSpher(point_old_coord);

	// Action = 24ti
	if (action_space == 24) {
		spher_coord.z = (0.78539*(rand() / float(RAND_MAX)) - 0.39269) + spher_coord.z;		// add or subtract randomly range {-22.5, 22.5} degrees to phi, in radian
		if (point_old_coord.z < 0.33) {
			spher_coord.y = 0.33*(rand() / float(RAND_MAX)) + 1.23;		// math done on the notes: theta - 0.168 < theta < theta - 0.168
		}
		else if (point_old_coord.z >= 0.33 && point_old_coord.z < 0.66) {
			spher_coord.y = 0.389*(rand() / float(RAND_MAX)) + 0.841;		// theta - 0.192 < theta < theta - 0.192
		}
		else {
			spher_coord.y = 0.841*(rand() / float(RAND_MAX));			//theta - 0.42 < theta < theta - 0.42
		}
	}	   	 

	// Action = 72, Original action space (equally sized patches)
	if (action_space == 72 && equally_sized_patches) {
		spher_coord.z = (0.523*(rand() / float(RAND_MAX)) - 0.261) + spher_coord.z;		// Action 72
		if (point_old_coord.z < 0.167) {
			spher_coord.y = 0.16*(rand() / float(RAND_MAX)) + 1.40;		// math done on the notes: theta - 0.168 < theta < theta - 0.168
		}
		else if (point_old_coord.z >= 0.167 && point_old_coord.z < 0.33) {
			spher_coord.y = 0.169*(rand() / float(RAND_MAX)) + 1.23;		// theta - 0.192 < theta < theta - 0.192
		}
		else if (point_old_coord.z >= 0.33 && point_old_coord.z < 0.5) {
			spher_coord.y = 0.182*(rand() / float(RAND_MAX)) + 1.048;		// theta - 0.192 < theta < theta - 0.192
		}
		else if (point_old_coord.z >= 0.5 && point_old_coord.z < 0.66) {
			spher_coord.y = 0.207*(rand() / float(RAND_MAX)) + 0.841;		// theta - 0.192 < theta < theta - 0.192
		}
		else if (point_old_coord.z >= 0.66 && point_old_coord.z < 0.833) {
			spher_coord.y = 0.317*(rand() / float(RAND_MAX)) + 0.523;		// theta - 0.192 < theta < theta - 0.192
		}
		else {
			spher_coord.y = 0.523*(rand() / float(RAND_MAX));
		}
	}
	// Action = 72, New action space (not-equally sized patches)
	else if (action_space == 72 && not_equally_sized_patches) {
		spher_coord.z = (0.523*(rand() / float(RAND_MAX)) - 0.261) + spher_coord.z;		// Action 72
		if (point_old_coord.z < 0.408) {
			spher_coord.y = 0.42*(rand() / float(RAND_MAX)) + 1.15;    // math done on the notes: theta - 0.168 < theta < theta - 0.168
		}
		else if (point_old_coord.z >= 0.408 && point_old_coord.z < 0.577) {
			spher_coord.y = 0.195*(rand() / float(RAND_MAX)) + 0.955;		// theta - 0.192 < theta < theta - 0.192
		}
		else if (point_old_coord.z >= 0.577 && point_old_coord.z < 0.707) {
			spher_coord.y = 0.170*(rand() / float(RAND_MAX)) + 0.785;		// theta - 0.192 < theta < theta - 0.192
		}
		else if (point_old_coord.z >= 0.707 && point_old_coord.z < 0.816) {
			spher_coord.y = 0.170*(rand() / float(RAND_MAX)) + 0.615;		// theta - 0.192 < theta < theta - 0.192
		}
		else if (point_old_coord.z >= 0.816 && point_old_coord.z < 0.913) {
			spher_coord.y = 0.195*(rand() / float(RAND_MAX)) + 0.420;		// theta - 0.192 < theta < theta - 0.192
		}
		else {
			spher_coord.y = 0.42*(rand() / float(RAND_MAX));
		}
	}
	// Action = 72, Double Action depth
	else if (action_space == 72 && double_action_space) {
		int* x = getRowColumn(double_action);

		float w = 0.523;
		float h = getHeight(action);
		float w_4 = w / 4;

		float A = acos(std::max((point_old_coord.z - h / 2), (float)0));
		float B = acos(std::max((point_old_coord.z - h / 6), (float)0));
		float C = acos(std::min((point_old_coord.z + h / 6), (float)1));
		float D = acos(std::min((point_old_coord.z + h / 2), (float)1));

		if (x[0] == 0) {
			spher_coord.y = B + (rand() / float(RAND_MAX))*(A - B);
		}
		else if (x[0] == 1) {
			spher_coord.y = C + (rand() / float(RAND_MAX))*(B - C);
		}
		else if (x[0] == 2) {
			spher_coord.y = D + (rand() / float(RAND_MAX))*(C - D);
		}
		if (x[1] == 0) {
			spher_coord.z = (spher_coord.z - w / 2) + (rand() / float(RAND_MAX)) * w_4;
		}
		else if (x[1] == 1) {
			spher_coord.z = (spher_coord.z - w_4) + (rand() / float(RAND_MAX)) * w_4;
		}
		else if (x[1] == 2) {
			spher_coord.z = spher_coord.z + (rand() / float(RAND_MAX)) * w_4;
		}
		else if (x[1] == 3) {
			spher_coord.z = (spher_coord.z + w_4) + (rand() / float(RAND_MAX)) * w_4;
		}
	}

	point_old_coord = spherToCart(spher_coord);
	return (u*point_old_coord.x + v * point_old_coord.y + w * point_old_coord.z); // new_point.x * u + new_point.y * v + new_point.z * w + hitting_point
	//return (v*point_old_coord.x + w * point_old_coord.y + u * point_old_coord.z); // new_point.x * u + new_point.y * v + new_point.z * w + hitting_point
}

int get_proportional_action(py::array_t<float> array_prob, int size_arr){
	float total = 0;
	py::buffer_info buf1 = array_prob.request();
	float *ptr1 = (float *)buf1.ptr;
	for (int i = 0; i < size_arr; i++) {
		total += ptr1[i];
	}
	
	float prob = rand() / float(RAND_MAX) * 0.9999;
	float cumulativeProbability = 0.0;
	int action = 0;
	for (int i = 0; i < action_space; i++) {
		cumulativeProbability += ptr1[i]/total;
		if (prob <= cumulativeProbability) {
			action = i;
			break;
		}
	}
	return action;	
}

float cumulative_q(std::map<Action, Direction> *dictAction, Vec &nl, py::array_t<float> array_prob) {
	std::map<Action, Direction> &addrDictAction = *dictAction;

	float cumulative_q = 0;
	py::buffer_info buf1 = array_prob.request();
	float *ptr1 = (float *)buf1.ptr;

	// Create temporary coordinates system
	Vec w = nl.norm();
	const Vec& u = getTangent(w).norm();
	const Vec& v = (w.cross(u)).norm();

	for (int i = 0; i < action_space; i++) {
		// calculate cos_theta_i
		//Vec action_vector = Vec(u*addrDictAction[i].x + v * addrDictAction[i].y + w * addrDictAction[i].z).norm();
		//const float& cos_theta_i = w.dot(action_vector);
		//cumulative_q += dict_next_state[i] * addrDictAction[i].z;
		cumulative_q += ptr1[i] * addrDictAction[i].z;
	}
	return cumulative_q;
}

PYBIND11_MODULE(smallpt_pybind, m) {
	py::class_<Vec>(m, "Vec")
		.def(py::init<float, float, float>(), py::arg("x") = 0, py::arg("y") = 0, py::arg("z") = 0)
		.def(py::self + py::self)
		.def(py::self - py::self)
		.def(py::self * float())
		.def(py::self * double())
		.def(py::self * int())
		.def("mult", &Vec::mult)
		.def("norm", &Vec::norm)
		.def("dot", &Vec::dot)
		.def("magnitude", &Vec::magnitude)
		.def("get_max", &Vec::get_max)
		.def("get_x", &Vec::get_x)
		.def("get_y", &Vec::get_y)
		.def("get_z", &Vec::get_z)
		.def("cross", &Vec::cross);

	py::class_<Ray>(m, "Ray")
		.def(py::init<Vec, Vec>())
		.def_readwrite("o", &Ray::o)
		.def_readwrite("d", &Ray::d);
	
	py::class_<Hitable, PyHitable> hitable(m, "Hitable");
	hitable.
		def(py::init<>())
		.def("intersect", &Hitable::intersect)
		.def("normal", &Hitable::normal)
		.def("get_e", &Hitable::get_e)
		.def("get_c", &Hitable::get_c);
	

	py::class_<Hit_record>(m, "Hit_record")
		.def(py::init<>())
		.def("get_id", &Hit_record::get_id)
		.def("get_oldid", &Hit_record::get_oldid)
		.def("get_t", &Hit_record::get_t)
		.def("get_hit", &Hit_record::get_hit)
		.def("set_id", &Hit_record::set_id)
		.def("set_oldid", &Hit_record::set_oldid)
		.def("set_hit", &Hit_record::set_hit)
		.def("set_t", &Hit_record::set_t)
		.def("get_nl", &Hit_record::get_nl)
		.def("get_e", &Hit_record::get_e)
		.def("get_c", &Hit_record::get_c)
		.def("get_prob", &Hit_record::get_prob)
		.def("get_oldstate", &Hit_record::get_oldstate)
		.def("get_oldaction", &Hit_record::get_oldaction)
		.def("set_prob", &Hit_record::set_prob)
		.def("get_BRDF", &Hit_record::get_BRDF)
		.def("set_BRDF", &Hit_record::set_BRDF)
		.def("set_costheta", &Hit_record::set_costheta)
		.def("get_costheta", &Hit_record::get_costheta)
		.def("set_oldaction", &Hit_record::set_oldaction);
	
	py::class_<Rectangle_xz>(m, "Rectangle_xz", hitable)
		.def(py::init<float, float, float, float, float, Vec, Vec>())
		.def_readwrite("x1", &Rectangle_xz::x1)
		.def_readwrite("x2", &Rectangle_xz::x2)
		.def_readwrite("z1", &Rectangle_xz::z1)
		.def_readwrite("z2", &Rectangle_xz::z2)
		.def_readwrite("y", &Rectangle_xz::y)
		.def_readwrite("e", &Rectangle_xz::e)
		.def_readwrite("c", &Rectangle_xz::c)
		.def("intersect", &Rectangle_xz::intersect)
		.def("normal", &Rectangle_xz::normal)
		.def("get_c", &Rectangle_xz::get_c)
		.def("get_e", &Rectangle_xz::get_e);

	py::class_<Rectangle_xy>(m, "Rectangle_xy", hitable)
		.def(py::init<float, float, float, float, float, Vec, Vec>())
		.def_readwrite("x1", &Rectangle_xy::x1)
		.def_readwrite("x2", &Rectangle_xy::x2)
		.def_readwrite("z1", &Rectangle_xy::y1)
		.def_readwrite("z2", &Rectangle_xy::y2)
		.def_readwrite("y", &Rectangle_xy::z)
		.def_readwrite("e", &Rectangle_xy::e)
		.def_readwrite("c", &Rectangle_xy::c)
		.def("intersect", &Rectangle_xy::intersect)
		.def("normal", &Rectangle_xy::normal)
		.def("get_c", &Rectangle_xy::get_c)
		.def("get_e", &Rectangle_xy::get_e);

	py::class_<Rectangle_yz>(m, "Rectangle_yz", hitable)
		.def(py::init<float, float, float, float, float, Vec, Vec>())
		.def_readwrite("x1", &Rectangle_yz::y1)
		.def_readwrite("x2", &Rectangle_yz::y2)
		.def_readwrite("z1", &Rectangle_yz::z1)
		.def_readwrite("z2", &Rectangle_yz::z2)
		.def_readwrite("y", &Rectangle_yz::x)
		.def_readwrite("e", &Rectangle_yz::e)
		.def_readwrite("c", &Rectangle_yz::c)
		.def("intersect", &Rectangle_yz::intersect)
		.def("normal", &Rectangle_yz::normal)
		.def("get_c", &Rectangle_yz::get_c)
		.def("get_e", &Rectangle_yz::get_e);

	py::class_<Rectangle_tilted>(m, "Rectangle_tilted", hitable)
		.def(py::init<Vec, Vec, Vec, Vec, Vec>())
		.def_readwrite("p1", &Rectangle_tilted::p1)
		.def_readwrite("p2", &Rectangle_tilted::p2)
		.def_readwrite("p3", &Rectangle_tilted::p3)
		.def_readwrite("e", &Rectangle_tilted::e)
		.def_readwrite("c", &Rectangle_tilted::c)
		.def("intersect", &Rectangle_tilted::intersect)
		.def("normal", &Rectangle_tilted::normal)
		.def("get_c", &Rectangle_tilted::get_c)
		.def("get_e", &Rectangle_tilted::get_e);

	py::class_<Camera>(m, "Camera")
		.def(py::init<Vec, Vec, Vec, float, float>())
		.def("get_ray", &Camera::get_ray);
		   
	m.def("importanceSampling_scattering", &importanceSampling_scattering);
	m.def("clamp", &clamp);
	m.def("intersection", &intersection);
	m.def("hittingPoint", &hittingPoint);
	m.def("spherToCart", &spherToCart);
	m.def("cartToSpher", &cartToSpher);
	m.def("initialize_dictAction", &initialize_dictAction);
	m.def("getTangent", &getTangent);
	m.def("DQNScattering", &DQNScattering);
	m.def("get_proportional_action", &get_proportional_action);
	m.def("cumulative_q", &cumulative_q);
	m.def("getHeight", &getHeight);
	m.def("getRowColumn", &getRowColumn);

}