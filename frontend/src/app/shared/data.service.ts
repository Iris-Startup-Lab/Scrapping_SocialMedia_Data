import { Injectable } from '@angular/core';
// Añadiendo extras para el servicio de datos
import { HttpClient } from '@angular/common/http';
//import { HttpClientModule } from '@angular/common/http';
import { Observable } from 'rxjs';


@Injectable({
  providedIn: 'root'
})

export class DataService {
  private apiUrl = 'http://127.0.0.1:8000/api/';

  constructor(private http: HttpClient) { }

  getTweets(): Observable<any[]> {
    console.log('Llamando a la API para obtener tweets...'); // Para debuggeo
    //return this.http.get<any[]>(this.apiUrl + 'datos-web/'); 
    return this.http.get<any[]>(this.apiUrl + 'tweet/'); 
    //return this.http.get<any[]>(this.apiUrl + 'resultados-analisis/'); // Cambiado a 'tweets/' para que coincida con la API
  }

  getSentimentResults(tweetIds: string[]): Observable<any[]> {
    console.log('Llamando a la API para obtener resultados de sentimiento dentro de getSentimentResults...'); 
    return this.http.post<any[]>(this.apiUrl + 'resultados-analisis-tweets/', { tweet_ids: tweetIds });
  }
}

console.log('Pasó el data service');