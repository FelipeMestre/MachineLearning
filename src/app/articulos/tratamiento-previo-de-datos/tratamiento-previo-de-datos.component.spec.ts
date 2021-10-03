import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TratamientoPrevioDeDatosComponent } from './tratamiento-previo-de-datos.component';

describe('TratamientoPrevioDeDatosComponent', () => {
  let component: TratamientoPrevioDeDatosComponent;
  let fixture: ComponentFixture<TratamientoPrevioDeDatosComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ TratamientoPrevioDeDatosComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(TratamientoPrevioDeDatosComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
