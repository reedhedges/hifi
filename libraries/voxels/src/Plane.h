//
//  Plane.h
//  hifi
//
//  Created by Brad Hefta-Gaub on 04/11/13.
//  Originally from lighthouse3d. Modified to utilize glm::vec3 and clean up to our coding standards
//
//  Simple plane class.
//

#ifndef _PLANE_
#define _PLANE_

#include <glm/glm.hpp>

class Plane  
{
public:
	Plane(const glm::vec3 &v1, const glm::vec3 &v2, const glm::vec3 &v3) { set3Points(v1,v2,v3); }
	Plane() : _normal(0,0,0), _point(0,0,0), _dCoefficient(0) {};
	~Plane() {} ;

    // methods for defining the plane
	void set3Points(const glm::vec3 &v1, const glm::vec3 &v2, const glm::vec3 &v3);
	void setNormalAndPoint(const glm::vec3 &normal, const glm::vec3 &point);
	void setCoefficients(float a, float b, float c, float d);

    // getters	
	const glm::vec3& getNormal() const { return _normal; };
	const glm::vec3& getPoint() const { return _point; };
	float getDCoefficient() const { return _dCoefficient; };

    // utilities
	float distance(const glm::vec3 &point) const;
	void print() const;

private:
	glm::vec3 _normal;
	glm::vec3 _point;
	float _dCoefficient;
};


#endif