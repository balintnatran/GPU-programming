__kernel void update(__global float2 *v, __global float2 *p, __global float* m, float dt)
{
    int gid = get_global_id(0);

    float2 a_gid;
    a_gid.x = 0;
    a_gid.y = 0;

    float mass = m[gid];
    float2 p_gid = p[gid];
    const float G = 0.0001;
	  dt = 0.004;

	for (int i = 0; i < get_global_size(0); ++i)
	{
		if (i == gid) continue;
		//////////////////
		// 1. síknegyed //
		//////////////////

		// x ∈ (0, 0.5), y ∈ (0, 0.5)
		if (p[i].x > 0 && p[i].x < 0.5 && p[i].y > 0 && p[i].y < 0.5 &&
			  p_gid.x > -0.5 && p_gid.x < 1 && p_gid.y > -0.5 && p_gid.y < 1)
		{
				float2 F = p_gid - p[i];
				float r2 = dot(F, F) + 0.006f;

				a_gid -= F * ((G * m[i] / r2) / sqrt(r2));
		}

		// x ∈ (0.5, 1), y ∈ (0, 0.5)
		if (p[i].x > 0.5 && p[i].x < 1 && p[i].y > 0 && p[i].y < 0.5 &&
		  	p_gid.x > 0 && p_gid.x < 1 && p_gid.y > -0.5 && p_gid.y < 1)
		{
			float2 F = p_gid - p[i];
			float r2 = dot(F, F) + 0.006f;

			a_gid -= F * ((G * m[i] / r2) / sqrt(r2));
		}

		// x ∈ (0, 0.5), y ∈ (0.5, 1)
		if (p[i].x > 0 && p[i].x < 0.5 && p[i].y > 0.5 && p[i].y < 1 &&
			  p_gid.x > -0.5 && p_gid.x < 1 && p_gid.y > 0 && p_gid.y < 1)
		{
			float2 F = p_gid - p[i];
			float r2 = dot(F, F) + 0.006f;

			a_gid -= F * ((G * m[i] / r2) / sqrt(r2));
		}

		// x ∈ (0.5, 1), y ∈ (0.5, 1)
		if (p[i].x > 0.5 && p[i].x < 1 && p[i].y > 0.5 && p[i].y < 1 &&
			  p_gid.x > 0 && p_gid.x < 1 && p_gid.y > 0 && p_gid.y < 1)
		{
			float2 F = p_gid - p[i];
			float r2 = dot(F, F) + 0.006f;

			a_gid -= F * ((G * m[i] / r2) / sqrt(r2));
		}

		//////////////////
		// 2. síknegyed //
		//////////////////

		// x ∈ (-0.5, 0), y ∈ (0, 0.5)
		if (p[i].x  < 0 && p[i].x  > -0.5 && p[i].y  > 0 && p[i].y  < 0.5 &&
			  p_gid.x < 0.5 && p_gid.x > -1 && p_gid.y > -0.5 && p_gid.y < 1)
		{
				float2 F = p_gid - p[i];
				float r2 = dot(F, F) + 0.006f;

				a_gid -= F * ((G * m[i] / r2) / sqrt(r2));
		}

		// x ∈ (-1, -0.5), y ∈ (0, 0.5)
		if (p[i].x  < -0.5 && p[i].x  > -1 && p[i].y  > 0 && p[i].y  < 0.5 &&
			  p_gid.x < 0 && p_gid.x > -1 && p_gid.y > -0.5 && p_gid.y < 1)
		{
			float2 F = p_gid - p[i];
			float r2 = dot(F, F) + 0.006f;

			a_gid -= F * ((G * m[i] / r2) / sqrt(r2));
		}

		// x ∈ (-0.5, 0), y ∈ (0.5, 1)
		if (p[i].x  < 0 && p[i].x  > -0.5 && p[i].y  > 0.5 && p[i].y  < 1 &&
			  p_gid.x < 0.5 && p_gid.x > -1 && p_gid.y > 0 && p_gid.y < 1)
		{
			float2 F = p_gid - p[i];
			float r2 = dot(F, F) + 0.006f;

			a_gid -= F * ((G * m[i] / r2) / sqrt(r2));
		}

		// x ∈ (-1, -0.5), y ∈ (0.5, 1)
		if (p[i].x  < -0.5 && p[i].x  > -1 && p[i].y  > 0.5 && p[i].y  < 1 &&
			  p_gid.x < 0 && p_gid.x > -1 && p_gid.y > 0 && p_gid.y < 1)
		{
			float2 F = p_gid - p[i];
			float r2 = dot(F, F) + 0.006f;

			a_gid -= F * ((G * m[i] / r2) / sqrt(r2));
		}

		//////////////////
		// 3. síknegyed //
		//////////////////

		// x ∈ (-0.5, 0), y ∈ (-0.5, 0)
		if (p[i].x  < 0 && p[i].x  > -0.5 && p[i].y  < 0 && p[i].y  > -0.5 &&
			  p_gid.x < 0.5 && p_gid.x > -1 && p_gid.y < 0.5 && p_gid.y > -1)
		{
				float2 F = p_gid - p[i];
				float r2 = dot(F, F) + 0.006f;

				a_gid -= F * ((G * m[i] / r2) / sqrt(r2));
		}

		// x ∈ (-1, -0.5), y ∈ (-0.5, 0)
		if (p[i].x  < -0.5 && p[i].x  > -1 && p[i].y  < 0 && p[i].y  > -0.5 &&
			  p_gid.x < 0 && p_gid.x > -1 && p_gid.y < 0.5 && p_gid.y > -1)
		{
			float2 F = p_gid - p[i];
			float r2 = dot(F, F) + 0.006f;

			a_gid -= F * ((G * m[i] / r2) / sqrt(r2));
		}

		// x ∈ (-0.5, 0), y ∈ (-1, -0.5)
		if (p[i].x  < 0 && p[i].x  > -0.5 && p[i].y  < -0.5 && p[i].y  > -1 &&
			  p_gid.x < 0.5 && p_gid.x > -1 && p_gid.y < 0 && p_gid.y > -1)
		{
			float2 F = p_gid - p[i];
			float r2 = dot(F, F) + 0.006f;

			a_gid -= F * ((G * m[i] / r2) / sqrt(r2));
		}

		// x ∈ (-1, -0.5), y ∈ (-1, -0.5)
		if (p[i].x  < -0.5 && p[i].x  > -1 && p[i].y  < -0.5 && p[i].y  > -1 &&
			  p_gid.x < 0 && p_gid.x > -1 && p_gid.y < 0 && p_gid.y > -1)
		{
			float2 F = p_gid - p[i];
			float r2 = dot(F, F) + 0.006f;

			a_gid -= F * ((G * m[i] / r2) / sqrt(r2));
		}

		//////////////////
		// 4. síknegyed //
		//////////////////

		// x ∈ (0, 0.5), y ∈ (-0.5, 0)
		if (p[i].x  > 0 && p[i].x  < 0.5 && p[i].y  < 0 && p[i].y  > -0.5 &&
			  p_gid.x > -0.5 && p_gid.x < 1 && p_gid.y < 0.5 && p_gid.y > -1)
		{
			float2 F = p_gid - p[i];
			float r2 = dot(F, F) + 0.006f;

			a_gid -= F * ((G * m[i] / r2) / sqrt(r2));
		}

		// x ∈ (0.5, 1), y ∈ (-0.5, 0)
		if (p[i].x  > 0.5 && p[i].x  < 1 && p[i].y  < 0 && p[i].y  > -0.5 &&
			  p_gid.x > 0 && p_gid.x < 1 && p_gid.y < 0.5 && p_gid.y > -1)
		{
			float2 F = p_gid - p[i];
			float r2 = dot(F, F) + 0.006f;

			a_gid -= F * ((G * m[i] / r2) / sqrt(r2));
		}

		// x ∈ (0, 0.5), y ∈ (-1, -0.5)
		if (p[i].x  > 0 && p[i].x  < 0.5 && p[i].y  < -0.5 && p[i].y  > -1 &&
			  p_gid.x > -0.5 && p_gid.x < 1 && p_gid.y < 0 && p_gid.y > -1)
		{
			float2 F = p_gid - p[i];
			float r2 = dot(F, F) + 0.006f;

			a_gid -= F * ((G * m[i] / r2) / sqrt(r2));
		}

		// x ∈ (0.5, 1), y ∈ (-1, -0.5)
		if (p[i].x  > 0.5 && p[i].x  < 1 && p[i].y  < -0.5 && p[i].y  > -1 &&
			  p_gid.x > 0 && p_gid.x < 1 && p_gid.y < 0 && p_gid.y > -1)
		{
			float2 F = p_gid - p[i];
			float r2 = dot(F, F) + 0.006f;

			a_gid -= F * ((G * m[i] / r2) / sqrt(r2));
		}
	
    }

    float2 v_ = v[gid] + dt * a_gid; //"" v = a * t ""

    barrier(CLK_GLOBAL_MEM_FENCE);

    p[gid] = p_gid + dt * v_; // ""s = v * t""
    v[gid] = v_;
    
}
