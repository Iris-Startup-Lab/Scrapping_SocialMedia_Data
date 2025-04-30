/*
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class GeminiService {

  constructor() { }
}
*/

import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class GeminiService {
  private apiUrl = 'http://127.0.0.1:8000/api/'; 

  constructor(private http: HttpClient) { }

  getGeminiResponse(question: string): Observable<any> {
    return this.http.post(this.apiUrl + 'obtener-respuesta/', { pregunta: question });
  }
}